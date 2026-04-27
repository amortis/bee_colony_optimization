//! Classical ABC for TSP (port of `ABC_Classic/ABC_classic.py`).

use crate::tsp::{fitness_from_distance, DistanceMatrix};
use rand::seq::SliceRandom;
use rand::Rng;
use std::time::{Duration, Instant};

pub type Tour = Vec<usize>;

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

/// Order crossover (OX), same as Python `EmployedBee._crossover_with_partner`.
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

fn calc_selection_probs(employed: &[EmployedBee]) -> Vec<f64> {
    if employed.is_empty() {
        return Vec::new();
    }
    let max_f = employed
        .iter()
        .map(|b| b.fitness)
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = employed
        .iter()
        .map(|b| ((b.fitness - max_f) * 10.0).exp())
        .collect();
    let total: f64 = exp_vals.iter().sum();
    if total > 0.0 {
        exp_vals.iter().map(|e| e / total).collect()
    } else {
        vec![1.0 / employed.len() as f64; employed.len()]
    }
}

fn pick_weighted<R: Rng + ?Sized>(rng: &mut R, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    if weights.is_empty() {
        return 0;
    }
    if total <= 0.0 {
        return rng.gen_range(0..weights.len());
    }
    let r = rng.gen::<f64>() * total;
    let mut c = 0.0;
    for (i, w) in weights.iter().enumerate() {
        c += w;
        if r < c {
            return i;
        }
    }
    weights.len() - 1
}

fn mutate_tour<R: Rng + ?Sized>(rng: &mut R, base: &[usize]) -> Tour {
    let mut new_sol = base.to_vec();
    let n = new_sol.len();
    if n < 2 {
        return new_sol;
    }
    match rng.gen_range(0..3) {
        0 => {
            if n >= 2 {
                let i = rng.gen_range(0..n);
                let j = loop {
                    let x = rng.gen_range(0..n);
                    if x != i {
                        break x;
                    }
                };
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                new_sol[lo..=hi].reverse();
            }
        }
        1 => {
            let i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n);
            while i == j && n > 1 {
                j = rng.gen_range(0..n);
            }
            new_sol.swap(i, j);
        }
        _ => {
            let idx = rng.gen_range(0..n);
            let city = new_sol.remove(idx);
            let new_pos = rng.gen_range(0..new_sol.len().saturating_add(1));
            new_sol.insert(new_pos, city);
        }
    }
    new_sol
}

pub struct AbcClassic {
    dm: DistanceMatrix,
    num_employed: usize,
    num_onlooker: usize,
    limit: usize,
    patience: usize,
    employed_bees: Vec<EmployedBee>,
    pub best_solution: Option<Tour>,
    best_fitness: f64,
    wait: usize,
    best_iteration: usize,
    convergence_history: Vec<f64>,
    run_started_at: Option<Instant>,
    last_run_elapsed: Option<Duration>,
    last_run_iterations: Option<usize>,
}

impl AbcClassic {
    pub fn new(
        dm: DistanceMatrix,
        num_employed_bees: usize,
        num_onlooker_bees: usize,
        limit: usize,
        patience: usize,
    ) -> Self {
        Self {
            dm,
            num_employed: num_employed_bees,
            num_onlooker: num_onlooker_bees,
            limit,
            patience,
            employed_bees: Vec::new(),
            best_solution: None,
            best_fitness: f64::NEG_INFINITY,
            wait: 0,
            best_iteration: 0,
            convergence_history: Vec::new(),
            run_started_at: None,
            last_run_elapsed: None,
            last_run_iterations: None,
        }
    }

    pub fn last_run_iterations(&self) -> Option<usize> {
        self.last_run_iterations
    }

    pub fn last_run_elapsed(&self) -> Option<Duration> {
        self.last_run_elapsed
    }

    pub fn last_run_elapsed_secs(&self) -> Option<f64> {
        self.last_run_elapsed.map(|d| d.as_secs_f64())
    }

    fn tour_distance(&self, tour: &[usize]) -> f64 {
        self.dm.tour_length(tour)
    }

    fn initialize_population<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.employed_bees.clear();
        self.best_fitness = f64::NEG_INFINITY;
        self.best_solution = None;
        let n = self.dm.n();
        for _ in 0..self.num_employed {
            let sol = random_tour(rng, n);
            let fit = fitness_from_distance(self.tour_distance(&sol));
            let bee = EmployedBee::new(sol, fit);
            if bee.fitness > self.best_fitness {
                self.best_fitness = bee.fitness;
                self.best_solution = Some(bee.solution.clone());
            }
            self.employed_bees.push(bee);
        }
    }

    fn employed_phase<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let n = self.employed_bees.len();
        for i in 0..n {
            let new_sol = {
                let bees = &self.employed_bees;
                let partners: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                if partners.is_empty() {
                    continue;
                }
                let pj = *partners.choose(rng).unwrap();
                ox_child(rng, &bees[i].solution, &bees[pj].solution)
            };
            let new_fit = fitness_from_distance(self.tour_distance(&new_sol));
            let bee = &mut self.employed_bees[i];
            if new_fit > bee.fitness {
                bee.solution = new_sol;
                bee.fitness = new_fit;
                bee.trial = 0;
                if new_fit > self.best_fitness {
                    self.best_fitness = new_fit;
                    self.best_solution = Some(bee.solution.clone());
                }
            } else {
                bee.trial += 1;
            }
        }
    }

    fn onlooker_phase<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        if self.employed_bees.is_empty() {
            return;
        }
        let probs = calc_selection_probs(&self.employed_bees);
        for _ in 0..self.num_onlooker {
            let init = self.employed_bees.choose(rng).unwrap();
            let mut my_fit = init.fitness;

            let mut best_ref = &self.employed_bees[0];
            for _ in 0..3 {
                let idx = pick_weighted(rng, &probs);
                let b = &self.employed_bees[idx];
                if b.fitness > best_ref.fitness {
                    best_ref = b;
                }
            }
            let new_sol = mutate_tour(rng, &best_ref.solution);
            let new_fit = fitness_from_distance(self.tour_distance(&new_sol));
            if new_fit > my_fit {
                my_fit = new_fit;
                if my_fit > self.best_fitness {
                    self.best_fitness = my_fit;
                    self.best_solution = Some(new_sol);
                }
            }
        }
    }

    fn scout_phase<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let n = self.dm.n();
        for i in 0..self.employed_bees.len() {
            if self.employed_bees[i].trial > self.limit {
                let new_sol = random_tour(rng, n);
                let nf = fitness_from_distance(self.tour_distance(&new_sol));
                self.employed_bees[i] = EmployedBee::new(new_sol, nf);
                if nf > self.best_fitness {
                    self.best_fitness = nf;
                    self.best_solution = Some(self.employed_bees[i].solution.clone());
                }
            }
        }
    }

    pub fn run<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        max_iterations: usize,
        verbose: bool,
        log_interval: usize,
    ) -> (Tour, f64) {
        self.initialize_population(rng);
        self.wait = 0;
        self.best_iteration = 0;
        self.convergence_history.clear();
        self.run_started_at = Some(Instant::now());
        self.last_run_elapsed = None;
        self.last_run_iterations = None;

        for iteration in 0..max_iterations {
            let old_best = self.best_fitness;
            self.employed_phase(rng);
            self.onlooker_phase(rng);
            self.scout_phase(rng);

            let current_distance = self
                .best_solution
                .as_ref()
                .map(|s| self.tour_distance(s))
                .unwrap_or(f64::INFINITY);
            self.convergence_history.push(current_distance);

            if self.best_fitness > old_best {
                self.wait = 0;
                self.best_iteration = iteration;
            } else {
                self.wait += 1;
            }

            self.last_run_iterations = Some(iteration + 1);

            if self.wait >= self.patience {
                if verbose {
                    eprintln!("Early stop at iteration {iteration}");
                }
                break;
            }

            if verbose && log_interval > 0 && iteration % log_interval == 0 {
                let elapsed = self
                    .run_started_at
                    .map(|t| t.elapsed().as_secs_f64())
                    .unwrap_or(0.0);
                eprintln!(
                    "iter {:4} | time {:.1}s | distance {:.2}",
                    iteration, elapsed, current_distance
                );
            }
        }

        let elapsed = self
            .run_started_at
            .map(|t| t.elapsed())
            .unwrap_or_default();
        self.last_run_elapsed = Some(elapsed);

        let best_distance = self
            .best_solution
            .as_ref()
            .map(|s| self.tour_distance(s))
            .unwrap_or(f64::INFINITY);
        let tour = self.best_solution.clone().unwrap_or_default();

        eprintln!(
            "Finished. Best distance = {:.2}, time = {:.3} s, iterations = {}",
            best_distance,
            elapsed.as_secs_f64(),
            self.last_run_iterations.unwrap_or(0)
        );

        (tour, best_distance)
    }
}
