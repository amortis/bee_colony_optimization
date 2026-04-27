//! Hybrid ABC + ILS for TSP (`ABCTSPILS`).

use crate::init::{greedy_init, nearest_neighbor};
use crate::local_search::{
    build_nearest_neighbors, fast_2opt, fast_2opt_neighbors, local_search_3opt,
};
use crate::perturb::{double_bridge, multi_insert, perturbation_random_2opt};
use crate::tsp::{fitness_from_distance, DistanceMatrix};
use rand::seq::{IteratorRandom, SliceRandom};
use rand::thread_rng;
use rand::Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::time::{Duration, Instant};

pub type Tour = Vec<usize>;

#[derive(Clone, Copy, Debug)]
pub struct AbcConfig {
    pub num_employed_bees: usize,
    pub num_onlooker_bees: usize,
    pub limit: usize,
    pub patience: usize,
    pub local_search_interval: usize,
    pub heuristic_init_ratio: f64,
    pub optimal_known: Option<f64>,
    /// Parallel employed phase (rayon), same idea as Python `ThreadPoolExecutor`.
    pub use_parallel: bool,
    /// Thread count for parallel phase; `None` = available parallelism.
    pub num_workers: Option<usize>,
}

impl Default for AbcConfig {
    fn default() -> Self {
        Self {
            num_employed_bees: 50,
            num_onlooker_bees: 50,
            limit: 100,
            patience: 200,
            local_search_interval: 50,
            heuristic_init_ratio: 0.7,
            optimal_known: None,
            use_parallel: true,
            num_workers: None,
        }
    }
}

#[derive(Clone)]
struct EmployedBee {
    solution: Tour,
    fitness: f64,
    trial: usize,
}

impl EmployedBee {
    fn new(solution: Tour, fitness: f64) -> Self {
        Self {
            solution,
            fitness,
            trial: 0,
        }
    }
}

fn apply_2opt_inplace(
    dm: &DistanceMatrix,
    neighbors: Option<&[Vec<usize>]>,
    tour: &mut Vec<usize>,
) {
    match neighbors {
        Some(nbr) => fast_2opt_neighbors(dm, tour, nbr),
        None => *tour = fast_2opt(dm, tour),
    }
}

/// Employed phase step using a frozen population snapshot (matches Python parallel `ThreadPoolExecutor` idea).
fn employed_step_snapshot<R: Rng + ?Sized>(
    idx: usize,
    mut bee: EmployedBee,
    all_tours: &[Tour],
    dm: &DistanceMatrix,
    neighbors: Option<&[Vec<usize>]>,
    limit: usize,
    rng: &mut R,
) -> EmployedBee {
    let lim_th = limit / 3;
    let n_bees = all_tours.len();

    if bee.trial > lim_th {
        let mut pert = double_bridge(rng, &bee.solution);
        pert = perturbation_random_2opt(rng, &pert, 2);
        let nf = fitness_from_distance(dm.tour_length(&pert));
        if nf > bee.fitness {
            bee.solution = pert;
            bee.fitness = nf;
            bee.trial = 0;
        }
    }

    let partner_idx = (0..n_bees).filter(|&p| p != idx).choose(rng).unwrap_or(idx);
    let partner_sol = &all_tours[partner_idx];
    let new_sol = AbcTspIls::ox_child(rng, &bee.solution, partner_sol);
    let nf = fitness_from_distance(dm.tour_length(&new_sol));
    let improved = nf > bee.fitness;
    if improved {
        bee.solution = new_sol;
        bee.fitness = nf;
        bee.trial = 0;
        let mut opt = bee.solution.clone();
        apply_2opt_inplace(dm, neighbors, &mut opt);
        bee.solution = opt;
        bee.fitness = fitness_from_distance(dm.tour_length(&bee.solution));
    } else {
        bee.trial += 1;
    }
    bee
}

pub struct AbcTspIls {
    dm: DistanceMatrix,
    config: AbcConfig,
    neighbors: Option<Vec<Vec<usize>>>,
    employed: Vec<EmployedBee>,
    best_tour: Tour,
    best_distance: f64,
    history: Vec<f64>,
    wait: usize,
    best_iteration: usize,
    historical_best: Vec<Tour>,
    max_historical: usize,
    ls_improvement_history: Vec<f64>,
    edge_memory: Vec<f64>,
    memory_decay: f64,
    local_search_interval: usize,
    limit: usize,
    patience: usize,
    num_cities: usize,
    /// Set at the start of each `run()` immediately after `initialize_population` (same as Python `start_time`).
    run_started_at: Option<Instant>,
    /// Wall time of the last finished `run()` from `run_started_at` until return (main loop + final print).
    last_run_elapsed: Option<Duration>,
    /// How many main-loop iterations actually executed (1..=max_iterations).
    last_run_iterations: Option<usize>,
}

impl AbcTspIls {
    pub fn new(dm: DistanceMatrix, config: AbcConfig) -> Self {
        let n = dm.n();
        let n_neighbors = if n >= 500 {
            50
        } else if n >= 300 {
            40
        } else if n >= 200 {
            30
        } else {
            20
        };
        let neighbors = if n > 50 {
            Some(build_nearest_neighbors(&dm, n_neighbors))
        } else {
            None
        };

        let best_tour: Vec<usize> = (0..n).collect();
        Self {
            dm,
            config,
            neighbors,
            employed: Vec::new(),
            best_tour,
            best_distance: f64::INFINITY,
            history: Vec::new(),
            wait: 0,
            best_iteration: 0,
            historical_best: Vec::new(),
            max_historical: 5,
            ls_improvement_history: Vec::new(),
            edge_memory: vec![0.0; n * n],
            memory_decay: 0.95,
            local_search_interval: config.local_search_interval.max(1),
            limit: config.limit,
            patience: config.patience,
            num_cities: n,
            run_started_at: None,
            last_run_elapsed: None,
            last_run_iterations: None,
        }
    }

    /// Iterations completed in the last `run()` (including early stop).
    #[inline]
    pub fn last_run_iterations(&self) -> Option<usize> {
        self.last_run_iterations
    }

    /// Duration of the last completed `run()` (after population init, same interval as Python `get_elapsed_time`).
    #[inline]
    pub fn last_run_elapsed(&self) -> Option<Duration> {
        self.last_run_elapsed
    }

    #[inline]
    pub fn last_run_elapsed_secs(&self) -> Option<f64> {
        self.last_run_elapsed.map(|d| d.as_secs_f64())
    }

    /// Instant when the current/last `run()` started counting time (after init). `None` before first `run()`.
    #[inline]
    pub fn run_started_at(&self) -> Option<Instant> {
        self.run_started_at
    }

    fn fitness(&self, tour: &[usize]) -> f64 {
        fitness_from_distance(self.dm.tour_length(tour))
    }

    fn update_edge_memory(&mut self, tour: &[usize]) {
        if self.best_distance <= 0.0 {
            return;
        }
        let n = self.dm.n();
        let w = 1.0 / (self.best_distance + 1e-9);
        for i in 0..tour.len() {
            let c1 = tour[i];
            let c2 = tour[(i + 1) % tour.len()];
            let nv = (self.edge_memory[c1 * n + c2] * self.memory_decay).max(w);
            self.edge_memory[c1 * n + c2] = nv;
            self.edge_memory[c2 * n + c1] = nv;
        }
    }

    fn memory_guided_nn<R: Rng + ?Sized>(&self, _rng: &mut R, start: usize) -> Tour {
        let n = self.dm.n();
        let mut tour = vec![start];
        let mut unvisited: Vec<bool> = vec![true; n];
        unvisited[start] = false;
        let mut current = start;

        while tour.len() < n {
            let mut best_j = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for j in 0..n {
                if !unvisited[j] {
                    continue;
                }
                let distance_score = 1.0 / (self.dm.get(current, j) + 1e-9);
                let memory_score = self.edge_memory[current * n + j];
                let combined = 0.7 * distance_score + 0.3 * memory_score;
                if combined > best_score {
                    best_score = combined;
                    best_j = j;
                }
            }
            tour.push(best_j);
            unvisited[best_j] = false;
            current = best_j;
        }
        tour
    }

    fn biased_random_tour<R: Rng + ?Sized>(&self, rng: &mut R) -> Tour {
        let n = self.dm.n();
        let mut tour: Vec<usize> = (0..n).collect();
        tour.shuffle(rng);
        let lim = (n / 10).max(1).min(5);
        for _ in 0..lim {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            if i == j {
                continue;
            }
            let ip = (i + n - 1) % n;
            let inext = (i + 1) % n;
            let jp = (j + n - 1) % n;
            let jnext = (j + 1) % n;
            let ci = tour[i];
            let cj = tour[j];
            let old_m = self.edge_memory[tour[ip] * n + ci]
                + self.edge_memory[ci * n + tour[inext]]
                + self.edge_memory[tour[jp] * n + cj]
                + self.edge_memory[cj * n + tour[jnext]];
            let new_m = self.edge_memory[tour[ip] * n + cj]
                + self.edge_memory[cj * n + tour[inext]]
                + self.edge_memory[tour[jp] * n + ci]
                + self.edge_memory[ci * n + tour[jnext]];
            if new_m > old_m {
                tour.swap(i, j);
            }
        }
        tour
    }

    /// Order crossover (OX), same semantics as Python `EmployedBee.generate_new_solution`.
    fn ox_child<R: Rng + ?Sized>(rng: &mut R, a: &[usize], b: &[usize]) -> Tour {
        let size = a.len();
        if size < 2 {
            return a.to_vec();
        }
        let mut child: Vec<Option<usize>> = vec![None; size];
        let start = rng.gen_range(0..size - 1);
        let end = rng.gen_range(start + 1..=(start + size / 2).min(size));
        for i in start..end {
            child[i] = Some(a[i]);
        }
        let mut ptr = 0usize;
        for i in 0..size {
            if child[i].is_none() {
                while ptr < size {
                    let v = b[ptr];
                    ptr += 1;
                    if !child.iter().any(|&c| c == Some(v)) {
                        child[i] = Some(v);
                        break;
                    }
                }
            }
        }
        child.into_iter().map(|x| x.unwrap_or(0)).collect()
    }

    fn random_tour<R: Rng + ?Sized>(rng: &mut R, n: usize) -> Tour {
        let mut t: Vec<usize> = (0..n).collect();
        t.shuffle(rng);
        t
    }

    fn initialize_population<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let n = self.dm.n();
        let total = self.config.num_employed_bees;
        let num_heuristic = (total as f64 * self.config.heuristic_init_ratio) as usize;
        let mut solutions: Vec<Tour> = Vec::new();

        for sol in &self.historical_best {
            if solutions.len() >= 3 {
                break;
            }
            solutions.push(sol.clone());
        }

        if num_heuristic > 0 {
            solutions.push(nearest_neighbor(rng, &self.dm, Some(0)));
        }

        let num_memory = num_heuristic / 3;
        for _ in 0..num_memory {
            let start = rng.gen_range(0..n);
            solutions.push(self.memory_guided_nn(rng, start));
        }

        while solutions.len() < num_heuristic {
            solutions.push(nearest_neighbor(rng, &self.dm, None));
        }

        solutions.push(greedy_init(&self.dm));

        let num_biased = ((total - solutions.len()) / 2).max(0);
        for _ in 0..num_biased {
            solutions.push(self.biased_random_tour(rng));
        }

        while solutions.len() < total {
            solutions.push(Self::random_tour(rng, n));
        }

        self.employed.clear();
        self.best_distance = f64::INFINITY;
        self.best_tour = Self::random_tour(rng, n);

        for sol in solutions.into_iter().take(total) {
            let fit = self.fitness(&sol);
            let d = self.dm.tour_length(&sol);
            if d < self.best_distance {
                self.best_distance = d;
                self.best_tour = sol.clone();
            }
            self.employed.push(EmployedBee::new(sol, fit));
        }

        self.update_edge_memory(&self.best_tour.clone());
    }

    fn apply_2opt_best(&mut self, tour: &mut Vec<usize>) {
        apply_2opt_inplace(&self.dm, self.neighbors.as_deref(), tour);
    }

    fn employed_phase<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let n_bees = self.employed.len();
        if n_bees == 0 {
            return;
        }

        let neighbors_slice = self.neighbors.as_deref();
        let use_parallel = self.config.use_parallel && n_bees > 10;

        if use_parallel {
            let bees_snapshot: Vec<EmployedBee> = self.employed.clone();
            let all_tours: Vec<Tour> = bees_snapshot.iter().map(|b| b.solution.clone()).collect();
            let dm = &self.dm;
            let limit = self.limit;
            let threads = self.config.num_workers.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|x| x.get())
                    .unwrap_or(4)
            });

            let new_bees: Vec<EmployedBee> =
                match ThreadPoolBuilder::new().num_threads(threads.max(1)).build() {
                    Ok(pool) => pool.install(|| {
                        (0..n_bees)
                            .into_par_iter()
                            .map(|idx| {
                                let mut trng = thread_rng();
                                employed_step_snapshot(
                                    idx,
                                    bees_snapshot[idx].clone(),
                                    &all_tours,
                                    dm,
                                    neighbors_slice,
                                    limit,
                                    &mut trng,
                                )
                            })
                            .collect()
                    }),
                    Err(_) => (0..n_bees)
                        .map(|idx| {
                            let mut trng = thread_rng();
                            employed_step_snapshot(
                                idx,
                                bees_snapshot[idx].clone(),
                                &all_tours,
                                dm,
                                neighbors_slice,
                                limit,
                                &mut trng,
                            )
                        })
                        .collect(),
                };

            self.employed = new_bees;
            for b in &self.employed {
                let tl = self.dm.tour_length(&b.solution);
                if tl < self.best_distance {
                    self.best_distance = tl;
                    self.best_tour = b.solution.clone();
                }
            }
            return;
        }

        // Sequential: partners see updates from earlier indices in this iteration.
        let lim_th = self.limit / 3;
        for idx in 0..n_bees {
            if self.employed[idx].trial > lim_th {
                let mut pert = double_bridge(rng, &self.employed[idx].solution);
                pert = perturbation_random_2opt(rng, &pert, 2);
                let nf = self.fitness(&pert);
                if nf > self.employed[idx].fitness {
                    self.employed[idx].solution = pert;
                    self.employed[idx].fitness = nf;
                    self.employed[idx].trial = 0;
                }
            }

            let partner_idx = (0..n_bees).filter(|&p| p != idx).choose(rng).unwrap_or(idx);
            let partner_sol = self.employed[partner_idx].solution.clone();

            let new_sol = Self::ox_child(rng, &self.employed[idx].solution, &partner_sol);
            let nf = self.fitness(&new_sol);
            let improved = nf > self.employed[idx].fitness;
            if improved {
                self.employed[idx].solution = new_sol;
                self.employed[idx].fitness = nf;
                self.employed[idx].trial = 0;

                let mut opt = self.employed[idx].solution.clone();
                self.apply_2opt_best(&mut opt);
                self.employed[idx].solution = opt.clone();
                self.employed[idx].fitness = self.fitness(&opt);

                let tl = self.dm.tour_length(&self.employed[idx].solution);
                if tl < self.best_distance {
                    self.best_distance = tl;
                    self.best_tour = self.employed[idx].solution.clone();
                }
            } else {
                self.employed[idx].trial += 1;
            }
        }
    }

    fn onlooker_probs(&self, tours: &[Tour]) -> Vec<f64> {
        let fv: Vec<f64> = tours.iter().map(|t| self.fitness(t)).collect();
        let max_f = fv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = fv.iter().map(|f| ((f - max_f) * 10.0).exp()).sum();
        if exp_sum > 0.0 {
            fv.iter()
                .map(|f| ((f - max_f) * 10.0).exp() / exp_sum)
                .collect()
        } else {
            vec![1.0 / tours.len() as f64; tours.len()]
        }
    }

    fn onlooker_mutate<R: Rng + ?Sized>(rng: &mut R, base: &[usize]) -> Tour {
        let mut new_sol = base.to_vec();
        match rng.gen_range(0..3) {
            0 => {
                let mut ij: Vec<usize> = (0..new_sol.len()).choose_multiple(rng, 2).into_iter().collect();
                ij.sort_unstable();
                let (i, j) = (ij[0], ij[1]);
                new_sol[i..=j].reverse();
            }
            1 => {
                let ij: Vec<usize> = (0..new_sol.len()).choose_multiple(rng, 2).into_iter().collect();
                new_sol.swap(ij[0], ij[1]);
            }
            _ => {
                let city = new_sol.remove(rng.gen_range(0..new_sol.len()));
                new_sol.insert(rng.gen_range(0..=new_sol.len()), city);
            }
        }
        new_sol
    }

    fn select_base_tour<'a, R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        tours: &'a [Tour],
        probs: &[f64],
    ) -> &'a Tour {
        let i1 = random_choice_index(rng, probs);
        let i2 = random_choice_index(rng, probs);
        let i3 = random_choice_index(rng, probs);
        let a = self.fitness(&tours[i1]);
        let b = self.fitness(&tours[i2]);
        let c = self.fitness(&tours[i3]);
        if a >= b && a >= c {
            &tours[i1]
        } else if b >= c {
            &tours[i2]
        } else {
            &tours[i3]
        }
    }

    fn onlooker_phase<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let tours: Vec<Tour> = self.employed.iter().map(|b| b.solution.clone()).collect();
        if tours.is_empty() {
            return;
        }
        let probs = self.onlooker_probs(&tours);
        let mut best_onlooker_len = f64::INFINITY;
        let mut best_onlooker: Option<Tour> = None;

        for _ in 0..self.config.num_onlooker_bees {
            let mut sol = tours[rng.gen_range(0..tours.len())].clone();
            let mut fit = self.fitness(&sol);

            let base = self.select_base_tour(rng, &tours, &probs);
            let new_sol = Self::onlooker_mutate(rng, base);
            let nf = self.fitness(&new_sol);
            if nf > fit {
                sol = new_sol;
                fit = nf;
            }

            let tl = self.dm.tour_length(&sol);
            if tl < self.best_distance {
                self.best_distance = tl;
                self.best_tour = sol.clone();
            }
            if tl < best_onlooker_len {
                best_onlooker_len = tl;
                best_onlooker = Some(sol);
            }
        }

        if let Some(ref bot) = best_onlooker {
            let worst_e = (0..self.employed.len())
                .max_by(|&a, &b| {
                    self.dm
                        .tour_length(&self.employed[a].solution)
                        .partial_cmp(&self.dm.tour_length(&self.employed[b].solution))
                        .unwrap()
                })
                .unwrap_or(0);
            let worst_len = self.dm.tour_length(&self.employed[worst_e].solution);
            if best_onlooker_len + 1e-9 < worst_len {
                self.employed[worst_e].solution = bot.clone();
                self.employed[worst_e].fitness = self.fitness(bot);
                self.employed[worst_e].trial = 0;
            }
        }
    }

    fn scout_phase<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let best = self.best_tour.clone();
        let n = self.num_cities;
        for i in 0..self.employed.len() {
            if self.employed[i].trial > self.limit {
                let candidate = if n >= 200 {
                    let mut c = double_bridge(rng, &best);
                    c = multi_insert(rng, &c, 2);
                    c
                } else {
                    double_bridge(rng, &best)
                };
                let mut arr = candidate.clone();
                self.apply_2opt_best(&mut arr);
                self.employed[i].solution = arr.clone();
                self.employed[i].fitness = self.fitness(&arr);
                self.employed[i].trial = 0;
                let tl = self.dm.tour_length(&arr);
                if tl < self.best_distance {
                    self.best_distance = tl;
                    self.best_tour = arr;
                }
            }
        }
    }

    fn adapt_parameters(&mut self, iteration: usize, max_iterations: usize) {
        let progress = if max_iterations > 0 {
            iteration as f64 / max_iterations as f64
        } else {
            0.0
        };
        let base_interval = 50;
        self.local_search_interval = ((base_interval as f64) * (1.0 - progress * 0.5)).max(10.0) as usize;

        if self.history.len() >= 10 {
            let recent = self.history[self.history.len() - 10] - self.history[self.history.len() - 1];
            if recent < self.best_distance * 0.01 {
                self.patience = (self.patience + 10).min(500);
            } else {
                self.patience = (self.patience * 95 / 100).max(50);
            }
        }

        if !self.employed.is_empty() {
            let fv: Vec<f64> = self.employed.iter().map(|b| self.fitness(&b.solution)).collect();
            let mean = fv.iter().sum::<f64>() / fv.len() as f64;
            let var = fv.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / fv.len() as f64;
            if var > 0.0 {
                let vr = var / (mean + 1e-9);
                self.limit = (100.0 + 100.0 * (1.0 - vr)).max(50.0).min(200.0) as usize;
            }
        }
    }

    fn inject_diversity<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let best = self.best_tour.clone();
        let num = if self.num_cities >= 200 {
            (self.employed.len() / 4).max(1)
        } else {
            (self.employed.len() / 5).max(1)
        };
        for k in 0..num {
            let idx = (self.best_iteration + k) % self.employed.len();
            if self.num_cities >= 200 {
                if rng.gen::<f64>() < 0.8 {
                    let mut p = double_bridge(rng, &best);
                    p = multi_insert(rng, &p, 3);
                    self.employed[idx].solution = p;
                } else {
                    self.employed[idx].solution = Self::random_tour(rng, self.num_cities);
                }
            } else if rng.gen::<f64>() < 0.7 {
                self.employed[idx].solution = double_bridge(rng, &best);
            } else {
                self.employed[idx].solution = Self::random_tour(rng, self.num_cities);
            }
            self.employed[idx].fitness = self.fitness(&self.employed[idx].solution);
            self.employed[idx].trial = 0;
        }
    }

    fn adaptive_diversity_control<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        if self.employed.is_empty() {
            return;
        }
        let fv: Vec<f64> = self.employed.iter().map(|b| self.fitness(&b.solution)).collect();
        let mean = fv.iter().sum::<f64>() / fv.len() as f64;
        let var = fv.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / fv.len() as f64;

        let threshold = if self.num_cities >= 200 {
            self.best_distance * 0.008
        } else {
            self.best_distance * 0.01
        };
        if var < threshold || var < 1e-6 {
            self.inject_diversity(rng);
        }
        if self.best_distance > 0.0 {
            let vr = var / (self.best_distance + 1e-9);
            self.limit = (200.0 * (1.0 - vr.min(1.0))).max(50.0) as usize;
        }
    }

    fn force_light_perturbation<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let best = self.best_tour.clone();
        let num = (self.employed.len() / 3).max(2);
        let mut indices: Vec<usize> = (0..self.employed.len()).collect();
        indices.shuffle(rng);
        indices.truncate(num);

        for &idx in &indices {
            let mut pert = double_bridge(rng, &best);
            self.apply_2opt_best(&mut pert);
            self.employed[idx].solution = pert.clone();
            self.employed[idx].fitness = self.fitness(&pert);
            self.employed[idx].trial = 0;
            let new_len = self.dm.tour_length(&pert);
            if new_len < self.best_distance {
                self.best_distance = new_len;
                self.best_tour = pert;
            }
        }
    }

    fn global_kick<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let best_global = self.best_tour.clone();
        let mut best_idx = 0usize;
        let mut best_len = self.dm.tour_length(&self.employed[0].solution);
        for i in 1..self.employed.len() {
            let l = self.dm.tour_length(&self.employed[i].solution);
            if l < best_len {
                best_len = l;
                best_idx = i;
            }
        }

        for i in 0..self.employed.len() {
            if i == best_idx {
                continue;
            }
            let mut candidate = if self.num_cities >= 200 {
                let mut c = double_bridge(rng, &best_global);
                c = multi_insert(rng, &c, 2);
                c
            } else {
                double_bridge(rng, &best_global)
            };
            self.apply_2opt_best(&mut candidate);
            let new_len = self.dm.tour_length(&candidate);
            self.employed[i].solution = candidate.clone();
            self.employed[i].fitness = self.fitness(&candidate);
            self.employed[i].trial = 0;
            if new_len < self.best_distance {
                self.best_distance = new_len;
                self.best_tour = candidate;
            }
        }
    }

    fn adaptive_local_search(&mut self, tour: &[usize]) -> (Tour, f64) {
        let use_3opt = self.num_cities < 200
            && self.ls_improvement_history.len() > 10
            && self.ls_improvement_history.len() >= 5
            && {
                let s: f64 = self.ls_improvement_history[self.ls_improvement_history.len() - 5..]
                    .iter()
                    .sum::<f64>()
                    / 5.0;
                s < 0.1
            };

        if use_3opt {
            return local_search_3opt(&self.dm, tour);
        }
        if self.neighbors.is_some() {
            let mut t = tour.to_vec();
            self.apply_2opt_best(&mut t);
            let d = self.dm.tour_length(&t);
            return (t, d);
        }
        local_search_2opt_wrap(&self.dm, tour)
    }

    fn local_2opt_search_global<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let old = self.best_distance;
        let bt = self.best_tour.clone();
        let (improved_tour, improved_dist) = self.adaptive_local_search(&bt);
        let improvement = old - improved_dist;
        self.ls_improvement_history.push(improvement);
        if self.ls_improvement_history.len() > 20 {
            self.ls_improvement_history.remove(0);
        }

        if improved_dist + 1e-9 < self.best_distance {
            self.best_distance = improved_dist;
            self.best_tour = improved_tour.clone();
            if self.historical_best.len() >= self.max_historical {
                self.historical_best.remove(0);
            }
            self.historical_best.push(improved_tour.clone());
            self.update_edge_memory(&improved_tour);

            let num_elite = (self.employed.len() / 5).max(1);
            for i in 0..num_elite {
                let new_tour = if i == 0 {
                    improved_tour.clone()
                } else {
                    perturbation_random_2opt(rng, &improved_tour, 1)
                };
                if i < self.employed.len() {
                    self.employed[i].solution = new_tour;
                    self.employed[i].fitness = self.fitness(&self.employed[i].solution);
                    self.employed[i].trial = 0;
                }
            }
        }
    }

    /// Run hybrid ABC + ILS. Returns `(best tour, best distance)`.
    pub fn run<R: Rng + ?Sized>(&mut self, rng: &mut R, max_iterations: usize) -> (Tour, f64) {
        self.initialize_population(rng);
        let run_start = Instant::now();
        self.run_started_at = Some(run_start);
        self.wait = 0;
        self.best_iteration = 0;
        self.history.clear();

        let mut stagnation = 0usize;
        let mut completed_iterations = 0usize;

        for it in 0..max_iterations {
            let old_best = self.best_distance;

            self.employed_phase(rng);
            self.onlooker_phase(rng);
            self.scout_phase(rng);

            let base_interval = if self.num_cities >= 200 {
                (self.local_search_interval / 3).max(5)
            } else {
                (self.local_search_interval / 2).max(10)
            };
            if it > 0 && it % base_interval == 0 {
                self.local_2opt_search_global(rng);
            }

            self.history.push(self.best_distance);

            if stagnation > 10 && stagnation % 5 == 0 {
                self.force_light_perturbation(rng);
            }

            if self.best_distance + 1e-9 < old_best {
                self.wait = 0;
                self.best_iteration = it;
                stagnation = 0;
                self.update_edge_memory(&self.best_tour.clone());
            } else {
                self.wait += 1;
                stagnation += 1;
            }

            let kick_th = if self.num_cities >= 200 { 40 } else { 50 };
            if stagnation > kick_th {
                self.global_kick(rng);
                stagnation = 0;
            }

            if it > 0 {
                self.adapt_parameters(it, max_iterations);
            }

            let div_iv = if self.num_cities >= 200 { 15 } else { 20 };
            if it > 0 && it % div_iv == 0 {
                self.adaptive_diversity_control(rng);
            }
            if stagnation > 20 && it % 10 == 0 {
                self.adaptive_diversity_control(rng);
            }

            completed_iterations = it + 1;

            if self.wait >= self.patience {
                eprintln!(
                    "Early stopping at iteration {}, no improvement for {} iterations.",
                    it, self.patience
                );
                break;
            }

            if it % 50 == 0 {
                eprintln!(
                    "Iteration {}: best = {:.2}, stagnation = {}",
                    it, self.best_distance, stagnation
                );
            }
        }

        let elapsed = run_start.elapsed();
        self.last_run_elapsed = Some(elapsed);
        self.last_run_iterations = Some(completed_iterations);
        eprintln!(
            "Finished. Best distance = {:.2}, time = {:.3} s, iterations = {}",
            self.best_distance,
            elapsed.as_secs_f64(),
            completed_iterations
        );
        (self.best_tour.clone(), self.best_distance)
    }
}

fn random_choice_index<R: Rng + ?Sized>(rng: &mut R, probs: &[f64]) -> usize {
    let r: f64 = rng.gen();
    let mut c = 0.0;
    for (i, p) in probs.iter().enumerate() {
        c += p;
        if r <= c {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

fn local_search_2opt_wrap(dm: &DistanceMatrix, tour: &[usize]) -> (Tour, f64) {
    let t = fast_2opt(dm, tour);
    let d = dm.tour_length(&t);
    (t, d)
}
