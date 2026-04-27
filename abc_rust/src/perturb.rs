//! Perturbation operators (double-bridge, multi-insert, random 2-opt moves).

use rand::seq::index::sample;
use rand::Rng;

pub fn two_opt_swap(tour: &mut [usize], i: usize, k: usize) {
    let (lo, hi) = if i <= k { (i, k) } else { (k, i) };
    tour[lo..=hi].reverse();
}

pub fn double_bridge<R: Rng + ?Sized>(rng: &mut R, tour: &[usize]) -> Vec<usize> {
    let n = tour.len();
    if n < 8 {
        let mut t = tour.to_vec();
        let i = rng.gen_range(0..n);
        let k = rng.gen_range(0..n);
        let (i, k) = if i <= k { (i, k) } else { (k, i) };
        two_opt_swap(&mut t, i, k);
        return t;
    }
    let mut cuts: Vec<usize> = Vec::with_capacity(4);
    while cuts.len() < 4 {
        let x = rng.gen_range(1..n);
        if !cuts.contains(&x) {
            cuts.push(x);
        }
    }
    cuts.sort_unstable();
    let a = cuts[0];
    let b = cuts[1];
    let c = cuts[2];
    let d = cuts[3];

    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&tour[0..a]);
    out.extend_from_slice(&tour[b..c]);
    out.extend_from_slice(&tour[a..b]);
    out.extend_from_slice(&tour[c..d]);
    out.extend_from_slice(&tour[d..n]);
    out
}

/// Python-compatible multi-insert: removes random positions, then inserts block at random place.
pub fn multi_insert<R: Rng + ?Sized>(rng: &mut R, tour: &[usize], num_cities_to_move: usize) -> Vec<usize> {
    let n = tour.len();
    if n < num_cities_to_move + 2 {
        return tour.to_vec();
    }
    let num_to_move = (num_cities_to_move.min(n / 4)).max(1);
    let mut idx: Vec<usize> = sample(rng, n, num_to_move).into_iter().collect();
    idx.sort_unstable_by(|a, b| b.cmp(a));

    let mut new_tour: Vec<usize> = tour.to_vec();
    let mut cities_values: Vec<usize> = Vec::with_capacity(idx.len());
    for &i in &idx {
        cities_values.push(new_tour[i]);
    }
    for &i in &idx {
        new_tour.remove(i);
    }
    let insert_pos = rng.gen_range(0..=new_tour.len());
    for (k, city) in cities_values.into_iter().enumerate() {
        new_tour.insert(insert_pos + k, city);
    }
    new_tour
}

pub fn perturbation_random_2opt<R: Rng + ?Sized>(rng: &mut R, tour: &[usize], strength: usize) -> Vec<usize> {
    let n = tour.len();
    let mut new_tour = tour.to_vec();
    let s = strength.max(1);
    for _ in 0..s {
        let i = rng.gen_range(0..n);
        let k = rng.gen_range(0..n);
        let (i, k) = if i <= k { (i, k) } else { (k, i) };
        if i != k {
            two_opt_swap(&mut new_tour, i, k);
        }
    }
    new_tour
}
