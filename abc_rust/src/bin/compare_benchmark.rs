//! Сравнительный прогон: **ABC_Classic**, **ABCTSPILS**, **LKH** по всем `.tsp` в каталоге.
//! В CSV — сырые строки по каждому методу (без столбцов «сравнение» между методами).

use abc_rust::adaptive::adaptive_parameters;
use abc_rust::load_euc_2d;
use abc_rust::lkh::LkhRunner;
use abc_rust::task_sets::{BENCHMARK_TASKS, MENU_TASKS};
use abc_rust::{AbcClassic, AbcConfig, AbcTspIls};
use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::fs::File;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "ABC classic vs ABCTSPILS vs LKH — все задачи из каталога в один CSV")]
struct Args {
    #[arg(long, default_value = "../matrix_task")]
    matrix_dir: PathBuf,

    #[arg(long, default_value = "compare_benchmark_results.csv")]
    output_csv: PathBuf,

    #[arg(long, default_value_t = 42_u64)]
    seed_base: u64,

    /// Только один файл, напр. `a280.tsp`
    #[arg(long)]
    only: Option<String>,

    /// Путь к бинарнику LKH (как у `lkh-run`)
    #[arg(long)]
    lkh_bin: Option<PathBuf>,
}

#[derive(Clone)]
struct Row {
    task_file: String,
    optimal: Option<f64>,
    method: &'static str,
    best_distance: Option<f64>,
    time_sec: Option<f64>,
    iterations: Option<usize>,
    gap_percent: Option<f64>,
    status: &'static str,
    note: String,
}

fn build_optima_map() -> HashMap<String, f64> {
    let mut m = HashMap::new();
    for (f, o) in MENU_TASKS.iter().chain(BENCHMARK_TASKS.iter()) {
        m.entry((*f).to_string()).or_insert(*o);
    }
    m
}

fn default_lkh_path() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut p = cwd.clone();
    for _ in 0..12 {
        let c = p.join("LKH").join("LKH-2.0.7").join("LKH");
        if c.exists() {
            return c;
        }
        if !p.pop() {
            break;
        }
    }
    cwd.join("LKH").join("LKH-2.0.7").join("LKH")
}

fn list_tsp_files(dir: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut v: Vec<String> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "tsp").unwrap_or(false))
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    v.sort();
    Ok(v)
}

fn gap_pct(found: f64, opt: f64) -> f64 {
    if opt > 0.0 {
        (found - opt) / opt * 100.0
    } else {
        0.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let matrix_dir = Path::new(&args.matrix_dir);
    if !matrix_dir.is_dir() {
        eprintln!("Нет каталога: {}", matrix_dir.display());
        return Ok(());
    }

    let mut tasks = list_tsp_files(matrix_dir)?;
    if let Some(ref only) = args.only {
        let o = only.trim().to_string();
        tasks.retain(|t| t == &o);
        if tasks.is_empty() {
            eprintln!("Нет файла {o} в {}", matrix_dir.display());
            return Ok(());
        }
    }

    if tasks.is_empty() {
        eprintln!("В {} нет .tsp файлов", matrix_dir.display());
        return Ok(());
    }

    let optima = build_optima_map();
    let lkh_path = args.lkh_bin.unwrap_or_else(default_lkh_path);
    let lkh_runner = LkhRunner::new(&lkh_path);
    let lkh_ok = lkh_runner.binary_exists();
    if !lkh_ok {
        eprintln!(
            "Предупреждение: LKH не найден ({}) — строки LKH будут со статусом SKIP.",
            lkh_path.display()
        );
    }

    let work_dir = std::env::temp_dir().join(format!("lkh_compare_{}", std::process::id()));
    let _ = fs::remove_dir_all(&work_dir);
    fs::create_dir_all(&work_dir)?;

    let mut rows: Vec<Row> = Vec::new();

    for (ti, task_file) in tasks.iter().enumerate() {
        let path = matrix_dir.join(task_file);
        eprintln!("\n=== [{}] {} ===", ti + 1, task_file);

        let dm = match load_euc_2d(&path) {
            Ok(m) => m,
            Err(e) => {
                for method in ["ABC_Classic", "ABCTSPILS", "LKH"] {
                    rows.push(Row {
                        task_file: task_file.clone(),
                        optimal: None,
                        method,
                        best_distance: None,
                        time_sec: None,
                        iterations: None,
                        gap_percent: None,
                        status: "ERROR",
                        note: format!("load: {e}"),
                    });
                }
                continue;
            }
        };

        let n = dm.n();
        let p = adaptive_parameters(n);
        let optimal = optima.get(task_file.as_str()).copied();
        let seed = args.seed_base.wrapping_add(ti as u64 * 10_000);

        {
            let mut algo = AbcClassic::new(
                dm.clone(),
                p.num_employed_bees,
                p.num_onlooker_bees,
                p.limit,
                p.patience,
            );
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let (_tour, dist) = algo.run(&mut rng, p.max_iterations, false, 10_000);
            rows.push(make_row(
                task_file,
                optimal,
                "ABC_Classic",
                Some(dist),
                algo.last_run_elapsed_secs(),
                algo.last_run_iterations(),
                optimal.map(|o| gap_pct(dist, o)),
                "OK",
                String::new(),
            ));
        }

        {
            let config = AbcConfig {
                num_employed_bees: p.num_employed_bees,
                num_onlooker_bees: p.num_onlooker_bees,
                limit: p.limit,
                patience: p.patience,
                local_search_interval: p.local_search_interval,
                heuristic_init_ratio: p.heuristic_init_ratio,
                optimal_known: optimal,
                use_parallel: true,
                num_workers: Some(16),
            };
            let mut algo = AbcTspIls::new(dm.clone(), config);
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1));
            let (_t, dist) = algo.run(&mut rng, p.max_iterations);
            rows.push(make_row(
                task_file,
                optimal,
                "ABCTSPILS",
                Some(dist),
                algo.last_run_elapsed_secs(),
                algo.last_run_iterations(),
                optimal.map(|o| gap_pct(dist, o)),
                "OK",
                String::new(),
            ));
        }

        if !lkh_ok {
            rows.push(Row {
                task_file: task_file.clone(),
                optimal,
                method: "LKH",
                best_distance: None,
                time_sec: None,
                iterations: None,
                gap_percent: None,
                status: "SKIP",
                note: "LKH binary not found".to_string(),
            });
        } else {
            match lkh_runner.run(&path, &work_dir, 100_000, 120, 42) {
                Ok(r) => {
                    let dist = r.cost;
                    rows.push(make_row(
                        task_file,
                        optimal,
                        "LKH",
                        Some(dist),
                        Some(r.time_sec),
                        None,
                        optimal.map(|o| gap_pct(dist, o)),
                        "OK",
                        String::new(),
                    ));
                }
                Err(e) => {
                    rows.push(Row {
                        task_file: task_file.clone(),
                        optimal,
                        method: "LKH",
                        best_distance: None,
                        time_sec: None,
                        iterations: None,
                        gap_percent: None,
                        status: "ERROR",
                        note: e.to_string(),
                    });
                }
            }
        }
    }

    let _ = fs::remove_dir_all(&work_dir);

    write_csv(&args.output_csv, &rows)?;
    eprintln!("\nЗаписано: {}", args.output_csv.display());
    Ok(())
}

fn make_row(
    task_file: &str,
    optimal: Option<f64>,
    method: &'static str,
    best_distance: Option<f64>,
    time_sec: Option<f64>,
    iterations: Option<usize>,
    gap_percent: Option<f64>,
    status: &'static str,
    note: String,
) -> Row {
    Row {
        task_file: task_file.to_string(),
        optimal,
        method,
        best_distance,
        time_sec,
        iterations,
        gap_percent,
        status,
        note,
    }
}

fn write_csv(path: &Path, rows: &[Row]) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_writer(File::create(path)?);
    wtr.write_record([
        "task_file",
        "optimal",
        "method",
        "best_distance",
        "time_sec",
        "iterations",
        "gap_percent",
        "status",
        "note",
    ])?;
    for r in rows {
        wtr.write_record([
            &r.task_file,
            &r.optimal.map(|x| format!("{:.6}", x)).unwrap_or_default(),
            r.method,
            &r.best_distance
                .map(|x| format!("{:.6}", x))
                .unwrap_or_default(),
            &r.time_sec.map(|x| format!("{:.6}", x)).unwrap_or_default(),
            &r.iterations.map(|x| x.to_string()).unwrap_or_default(),
            &r.gap_percent
                .map(|x| format!("{:.6}", x))
                .unwrap_or_default(),
            r.status,
            &r.note,
        ])?;
    }
    wtr.flush()?;
    Ok(())
}
