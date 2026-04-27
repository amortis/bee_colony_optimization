//! 2-opt local search (full and neighbor-restricted).

use crate::perturb::two_opt_swap;
use crate::tsp::DistanceMatrix;

pub fn build_nearest_neighbors(dm: &DistanceMatrix, n_neighbors: usize) -> Vec<Vec<usize>> {
    let n = dm.n();
    let k = n_neighbors.min(n.saturating_sub(1)).max(1);
    let mut neighbors = vec![vec![0usize; k]; n];
    let mut order: Vec<usize> = (0..n).collect();

    for i in 0..n {
        order.sort_by(|&a, &b| {
            dm.get(i, a)
                .partial_cmp(&dm.get(i, b))
                .unwrap()
                .then_with(|| a.cmp(&b))
        });
        let mut w = 0;
        for &j in &order {
            if j == i {
                continue;
            }
            neighbors[i][w] = j;
            w += 1;
            if w >= k {
                break;
            }
        }
    }
    neighbors
}

/// First-improvement 2-opt (matches Numba `_fast_2opt_numba` structure).
pub fn fast_2opt(dm: &DistanceMatrix, tour: &[usize]) -> Vec<usize> {
    let n = dm.n();
    let mut t = tour.to_vec();
    let mut improved = true;
    while improved {
        improved = false;
        for i in 0..n.saturating_sub(1) {
            for j in (i + 2)..n {
                if j == n - 1 && i == 0 {
                    continue;
                }
                let a = t[i];
                let b = t[i + 1];
                let c = t[j];
                let d = t[(j + 1) % n];
                let delta = dm.get(a, c) + dm.get(b, d) - dm.get(a, b) - dm.get(c, d);
                if delta < -1e-9 {
                    t[i + 1..=j].reverse();
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }
    }
    t
}

/// Neighbor-restricted 2-opt (matches `_fast_2opt_neighbors_numba`).
pub fn fast_2opt_neighbors(dm: &DistanceMatrix, tour: &mut [usize], neighbors: &[Vec<usize>]) {
    let n = dm.n();
    let mut pos = vec![0usize; n];
    for i in 0..n {
        pos[tour[i]] = i;
    }

    let mut improved = true;
    while improved {
        improved = false;
        'outer: for i in 0..n {
            let u = tour[i];
            let u_next = tour[(i + 1) % n];
            let u_prev = tour[(i + n - 1) % n];

            for nb in &neighbors[u] {
                let v = *nb;
                if v == u_next || v == u_prev {
                    continue;
                }
                let j = pos[v];
                let v_next = tour[(j + 1) % n];

                let current_len = dm.get(u, u_next) + dm.get(v, v_next);
                let new_len = dm.get(u, v) + dm.get(u_next, v_next);

                if new_len < current_len - 1e-6 {
                    if j > i {
                        let mut low = i + 1;
                        let mut high = j;
                        while low < high {
                            tour.swap(low, high);
                            pos[tour[low]] = low;
                            pos[tour[high]] = high;
                            low += 1;
                            high -= 1;
                        }
                    } else {
                        continue;
                    }
                    improved = true;
                    break 'outer;
                }
            }
        }
    }
}

pub fn local_search_2opt(dm: &DistanceMatrix, initial: &[usize]) -> (Vec<usize>, f64) {
    let optimized = fast_2opt(dm, initial);
    let d = dm.tour_length(&optimized);
    (optimized, d)
}

/// Simplified 3-opt (same structure as Python `local_search_3opt` in `tsp_optimizations.py`):
/// evaluates one reconnection pattern, applies a 2-opt-style segment reversal when improvement found.
pub fn local_search_3opt(dm: &DistanceMatrix, initial: &[usize]) -> (Vec<usize>, f64) {
    let n = initial.len();
    if n < 4 {
        let d = dm.tour_length(initial);
        return (initial.to_vec(), d);
    }
    let mut current_tour = initial.to_vec();
    let mut improved = true;

    while improved {
        improved = false;
        let mut best_improvement = 0.0f64;
        let mut best_ij: Option<(usize, usize)> = None;

        for i in 0..n {
            for j in (i + 2)..n {
                for k in (j + 2)..n {
                    if k == n - 1 && i == 0 {
                        continue;
                    }
                    let a = current_tour[i];
                    let b = current_tour[(i + 1) % n];
                    let c = current_tour[j];
                    let d1 = current_tour[(j + 1) % n];
                    let e = current_tour[k];
                    let f = current_tour[(k + 1) % n];
                    let old_dist = dm.get(a, b) + dm.get(c, d1) + dm.get(e, f);
                    let new_dist = dm.get(a, d1) + dm.get(e, b) + dm.get(c, f);
                    let improvement = old_dist - new_dist;
                    if improvement > best_improvement {
                        best_improvement = improvement;
                        best_ij = Some((i, j));
                    }
                }
            }
        }

        if best_improvement > 0.0 {
            if let Some((i, j)) = best_ij {
                if j > i + 1 {
                    two_opt_swap(&mut current_tour, i, j);
                }
                improved = true;
            }
        }
    }

    let dist = dm.tour_length(&current_tour);
    (current_tour, dist)
}
