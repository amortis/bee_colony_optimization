//! CLI: load a TSPLIB `EUC_2D` instance and run ABC+ILS.

use abc_rust::load_euc_2d;
use abc_rust::{AbcClassic, AbcConfig, AbcTspIls};
use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "abc-tsp", about = "ABC + ILS for TSP (Rust port)")]
struct Args {
    /// Path to a `.tsp` file (EUC_2D, NODE_COORD_SECTION)
    #[arg(short, long)]
    tsp: PathBuf,

    #[arg(short, long, default_value_t = 500)]
    iterations: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 50)]
    employed: usize,

    #[arg(long, default_value_t = 50)]
    onlookers: usize,

    /// Known optimal tour length (for reporting gap), optional
    #[arg(long)]
    optimal: Option<f64>,

    /// Classical ABC (`ABC_Classic`) instead of hybrid ABCTSPILS
    #[arg(long)]
    classic: bool,

    /// Disable parallel employed phase (rayon)
    #[arg(long)]
    sequential: bool,

    #[arg(long)]
    workers: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let dm = load_euc_2d(&args.tsp)?;

    let (tour, dist, wall_sec) = if args.classic {
        let mut algo = AbcClassic::new(
            dm,
            args.employed,
            args.onlookers,
            20,
            100,
        );
        let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
        let (tour, dist) = algo.run(&mut rng, args.iterations, false, 50);
        (tour, dist, algo.last_run_elapsed_secs())
    } else {
        let config = AbcConfig {
            num_employed_bees: args.employed,
            num_onlooker_bees: args.onlookers,
            optimal_known: args.optimal,
            use_parallel: !args.sequential,
            num_workers: args.workers,
            ..Default::default()
        };
        let mut algo = AbcTspIls::new(dm, config);
        let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
        let (tour, dist) = algo.run(&mut rng, args.iterations);
        (tour, dist, algo.last_run_elapsed_secs())
    };

    println!("best_distance\t{:.6}", dist);
    if let Some(s) = wall_sec {
        println!("wall_time_sec\t{:.6}", s);
    }
    if let Some(opt) = args.optimal {
        let gap = 100.0 * (dist - opt) / opt;
        println!("optimal\t{:.6}", opt);
        println!("gap_percent\t{:.4}", gap);
    }
    println!(
        "tour\t{}",
        tour.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}
