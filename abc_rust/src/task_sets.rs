//! Known TSPLIB-style tasks + published optima (paths relative to `matrix_task/`).

/// Same list as repo root `main.py` (interactive menu).
pub const MENU_TASKS: &[(&str, f64)] = &[
    ("a280.tsp", 2579.0),
    ("eil51.tsp", 426.0),
    ("eil101.tsp", 629.0),
    ("lin318.tsp", 42029.0),
    ("pa561.tsp", 2763.0),
    ("pr1002.tsp", 259045.0),
    ("rat575.tsp", 6773.0),
    ("rd100.tsp", 7910.0),
    ("rd400.tsp", 15281.0),
    ("st70.tsp", 675.0),
];

/// Same list as repo root `benchmark.py`.
pub const BENCHMARK_TASKS: &[(&str, f64)] = &[
    ("a280.tsp", 2579.0),
    ("eil51.tsp", 426.0),
    ("eil101.tsp", 629.0),
    ("lin318.tsp", 42029.0),
    ("pa561.tsp", 2763.0),
    ("pr1002.tsp", 259045.0),
    ("rat575.tsp", 6773.0),
    ("rd100.tsp", 7910.0),
    ("rd400.tsp", 15281.0),
    ("st70.tsp", 675.0),
];
