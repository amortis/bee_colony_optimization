//! Construction heuristics (nearest neighbor, greedy).

use crate::tsp::DistanceMatrix;
use rand::Rng;

pub fn nearest_neighbor<R: Rng + ?Sized>(rng: &mut R, dm: &DistanceMatrix, start: Option<usize>) -> Vec<usize> {
    let n = dm.n();
    let start_city = start.unwrap_or_else(|| rng.gen_range(0..n));
    let mut tour = vec![start_city];
    let mut unvisited: Vec<bool> = vec![true; n];
    unvisited[start_city] = false;
    let mut current = start_city;

    while tour.len() < n {
        let mut best_j = None;
        let mut best_d = f64::INFINITY;
        for j in 0..n {
            if unvisited[j] {
                let d = dm.get(current, j);
                if d < best_d {
                    best_d = d;
                    best_j = Some(j);
                }
            }
        }
        if let Some(j) = best_j {
            tour.push(j);
            unvisited[j] = false;
            current = j;
        } else {
            break;
        }
    }
    tour
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            self.parent[ry] = rx;
        }
    }
}

/// Greedy edge insertion (same logic as Python `greedy_init`).
pub fn greedy_init(dm: &DistanceMatrix) -> Vec<usize> {
    let n = dm.n();
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n * n / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((dm.get(i, j), i, j));
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut degree = vec![0usize; n];
    let mut uf = UnionFind::new(n);
    let mut selected: Vec<(usize, usize)> = Vec::new();

    for &(_, u, v) in &edges {
        if degree[u] >= 2 || degree[v] >= 2 {
            continue;
        }
        let ru = uf.find(u);
        let rv = uf.find(v);
        if ru == rv && selected.len() < n - 1 {
            continue;
        }
        selected.push((u, v));
        degree[u] += 1;
        degree[v] += 1;
        uf.union(u, v);

        if selected.len() == n {
            break;
        }
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (u, v) in selected {
        adj[u].push(v);
        adj[v].push(u);
    }

    let start = (0..n).find(|&i| adj[i].len() == 1).unwrap_or(0);
    let mut tour = vec![start];
    let mut prev = usize::MAX;
    let mut current = start;
    for _ in 0..n - 1 {
        let neigh = &adj[current];
        let nxt = if neigh[0] != prev {
            neigh[0]
        } else {
            neigh[1]
        };
        tour.push(nxt);
        prev = current;
        current = nxt;
    }
    tour
}
