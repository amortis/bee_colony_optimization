//! Adaptive hyperparameters (same breakpoints as repo `main.py` / `get_adaptive_parameters`).

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveParams {
    pub num_employed_bees: usize,
    pub num_onlooker_bees: usize,
    pub limit: usize,
    pub patience: usize,
    pub local_search_interval: usize,
    pub heuristic_init_ratio: f64,
    pub max_iterations: usize,
}

/// Mirrors `get_adaptive_parameters` in `main.py`.
pub fn adaptive_parameters(num_cities: usize) -> AdaptiveParams {
    if num_cities < 100 {
        AdaptiveParams {
            num_employed_bees: 30,     
            num_onlooker_bees: 40,        
            limit: 150,                 
            patience: 700,               
            local_search_interval: 10,     
            heuristic_init_ratio: 0.80,    
            max_iterations: 2000,          
        }
    } else if num_cities < 200 {
        AdaptiveParams {
            num_employed_bees: 40,
            num_onlooker_bees: 50,
            limit: 200,
            patience: 1000,
            local_search_interval: 25,
            heuristic_init_ratio: 0.85,
            max_iterations: 5000,
        }
    } else if num_cities < 300 {
        AdaptiveParams {
            num_employed_bees: 50,
            num_onlooker_bees: 70,
            limit: 250,
            patience: 2000,
            local_search_interval: 20,
            heuristic_init_ratio: 0.9,
            max_iterations: 5000,
        }
    } else if num_cities < 500 {
        AdaptiveParams {
            num_employed_bees: 80,
            num_onlooker_bees: 100,
            limit: 100,
            patience: 3000,
            local_search_interval: 30,
            heuristic_init_ratio: 0.95,
            max_iterations: 10000,
        }
    } else {
        AdaptiveParams {
            num_employed_bees: (num_cities / 10).max(100),
            num_onlooker_bees: (num_cities / 8).max(120),
            limit: 80,
            patience: 4000,
            local_search_interval: 35,
            heuristic_init_ratio: 0.95,
            max_iterations: (num_cities * 15).max(15000),
        }
    }
}
