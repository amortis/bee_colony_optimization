//! Rust analogue of the repo root `benchmark.py`: multiple runs per task, CSV export.
//! GPU column is always `0` (no GPU in this port).

use abc_rust::adaptive::adaptive_parameters;
use abc_rust::load_euc_2d;
use abc_rust::task_sets::BENCHMARK_TASKS;
use abc_rust::{AbcConfig, AbcTspIls};
use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fs::File;
use std::path::{Path, PathBuf};

const NUM_RUNS: usize = 10;
const SUCCESS_TOLERANCE_PCT: f64 = 0.1;

#[derive(Parser, Debug)]
#[command(about = "Batch benchmark (see benchmark.py)")]
struct Args {
    /// Directory with `.tsp` files (e.g. `../matrix_task`)
    #[arg(long, default_value = "../matrix_task")]
    matrix_dir: PathBuf,

    #[arg(long, default_value = "benchmark_results_rust.csv")]
    output_csv: PathBuf,

    /// Run only one task basename (e.g. `a280.tsp`)
    #[arg(long)]
    only: Option<String>,

    #[arg(long, default_value_t = NUM_RUNS)]
    num_runs: usize,

    #[arg(long, default_value_t = 0)]
    seed_base: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let matrix_dir = Path::new(&args.matrix_dir);
    let mut all_rows: Vec<BenchRow> = Vec::new();

    let tasks: Vec<(&str, f64)> = if let Some(ref only) = args.only {
        let o = only.trim();
        BENCHMARK_TASKS.iter().copied().filter(|(f, _)| *f == o).collect()
    } else {
        BENCHMARK_TASKS.iter().copied().collect()
    };

    if tasks.is_empty() {
        eprintln!("No matching tasks for --only");
        return Ok(());
    }

    for (task_file, optimal) in tasks {
        let path = matrix_dir.join(task_file);
        eprintln!("=== {} (opt {}) ===", task_file, optimal);
        if !path.exists() {
            eprintln!("  skip: file not found {:?}", path);
            continue;
        }

        let dm = match load_euc_2d(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  skip load error: {e}");
                continue;
            }
        };
        let n = dm.n();
        let p = adaptive_parameters(n);

        let mut run_times: Vec<f64> = Vec::new();
        let mut run_dists: Vec<f64> = Vec::new();

        for run in 0..args.num_runs {
            let seed = args.seed_base + run as u64;
            let config = AbcConfig {
                num_employed_bees: p.num_employed_bees,
                num_onlooker_bees: p.num_onlooker_bees,
                limit: p.limit,
                patience: p.patience,
                local_search_interval: p.local_search_interval,
                heuristic_init_ratio: p.heuristic_init_ratio,
                optimal_known: Some(optimal),
                use_parallel: true,
                num_workers: Some(16),
            };
            let mut algo = AbcTspIls::new(dm.clone(), config);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);

            let t0 = std::time::Instant::now();
            let (_tour, best_dist) = algo.run(&mut rng, p.max_iterations);
            let elapsed = t0.elapsed().as_secs_f64();

            let gap_pct = if optimal > 0.0 {
                (best_dist - optimal) / optimal * 100.0
            } else {
                0.0
            };
            let success = gap_pct.abs() <= SUCCESS_TOLERANCE_PCT;
            eprintln!(
                "  run {}/{} | {:.1}s | dist {:.2} | gap {:+.3}% | {}",
                run + 1,
                args.num_runs,
                elapsed,
                best_dist,
                gap_pct,
                if success { "OK" } else { "FAIL" }
            );

            run_times.push(elapsed);
            run_dists.push(best_dist);

            all_rows.push(BenchRow {
                task: task_file.to_string(),
                run: run + 1,
                time_s: elapsed,
                distance: best_dist,
                optimal,
                gap_pct,
                success,
                avg_gpu_util: 0.0,
            });
        }

        if !run_times.is_empty() {
            let mean_t = run_times.iter().sum::<f64>() / run_times.len() as f64;
            let std_t = std_dev(&run_times);
            let mean_d = run_dists.iter().sum::<f64>() / run_dists.len() as f64;
            let std_d = std_dev(&run_dists);
            let succ = all_rows
                .iter()
                .filter(|r| r.task == task_file)
                .filter(|r| r.success)
                .count();
            eprintln!(
                "  summary: time {:.1}±{:.1}s | dist {:.2}±{:.2} | success {}/{}",
                mean_t,
                std_t,
                mean_d,
                std_d,
                succ,
                args.num_runs
            );
        }
    }

    write_csv(&args.output_csv, &all_rows)?;
    eprintln!("Wrote {}", args.output_csv.display());
    Ok(())
}

fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    v.sqrt()
}

struct BenchRow {
    task: String,
    run: usize,
    time_s: f64,
    distance: f64,
    optimal: f64,
    gap_pct: f64,
    success: bool,
    avg_gpu_util: f64,
}

fn write_csv(path: &Path, rows: &[BenchRow]) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_writer(File::create(path)?);
    wtr.write_record([
        "task",
        "run",
        "time_s",
        "distance",
        "optimal",
        "gap_pct",
        "success",
        "avg_gpu_util",
    ])?;
    for r in rows {
        wtr.write_record([
            &r.task,
            &r.run.to_string(),
            &format!("{:.6}", r.time_s),
            &format!("{:.6}", r.distance),
            &format!("{:.6}", r.optimal),
            &format!("{:.6}", r.gap_pct),
            &r.success.to_string(),
            &format!("{:.6}", r.avg_gpu_util),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}
