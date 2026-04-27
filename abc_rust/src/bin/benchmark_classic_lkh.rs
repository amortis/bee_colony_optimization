//! Бенчмарк: **ABC Classic** и **LKH** по фиксированному подмножеству задач (10 прогонов каждая).
//! Два отдельных CSV + в конце каждого файла блок итогов по задачам (русские заголовки).

use abc_rust::adaptive::adaptive_parameters;
use abc_rust::load_euc_2d;
use abc_rust::lkh::LkhRunner;
use abc_rust::AbcClassic;
use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Те же оптимумы, что в `benchmark.py` / `BENCHMARK_TASKS` для этих имён файлов.
const SUBSET_TASKS: &[(&str, f64)] = &[
    ("a280.tsp", 2579.0),
    ("rd100.tsp", 7910.0),
    ("rat575.tsp", 6773.0),
    ("pr1002.tsp", 259045.0),
];

const NUM_RUNS: usize = 10;
const SUCCESS_TOLERANCE_PCT: f64 = 0.1;

#[derive(Parser, Debug)]
#[command(about = "ABC Classic + LKH: 4 задачи, 10 прогонов, 2 CSV с итогами в конце")]
struct Args {
    #[arg(long, default_value = "../matrix_task")]
    matrix_dir: PathBuf,

    #[arg(long, default_value = "benchmark_classic_subset.csv")]
    output_classic_csv: PathBuf,

    #[arg(long, default_value = "benchmark_lkh_subset.csv")]
    output_lkh_csv: PathBuf,

    #[arg(long, default_value_t = 0_u64)]
    seed_base: u64,

    #[arg(long, default_value_t = NUM_RUNS)]
    num_runs: usize,

    #[arg(long)]
    lkh_bin: Option<PathBuf>,
}

#[derive(Clone)]
struct ClassicRow {
    task: String,
    run: usize,
    time_s: f64,
    distance: f64,
    optimal: f64,
    gap_pct: f64,
    success: bool,
}

#[derive(Clone)]
struct LkhRow {
    task: String,
    run: usize,
    time_s: Option<f64>,
    distance: Option<f64>,
    optimal: f64,
    gap_pct: Option<f64>,
    success: bool,
    status: &'static str,
    note: String,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let num_runs = args.num_runs.max(1);
    let matrix_dir = Path::new(&args.matrix_dir);

    let lkh_path = args.lkh_bin.unwrap_or_else(default_lkh_path);
    let lkh_runner = LkhRunner::new(&lkh_path);
    let lkh_ok = lkh_runner.binary_exists();
    if !lkh_ok {
        eprintln!(
            "Предупреждение: LKH не найден ({}). Строки LKH будут с ошибкой.",
            lkh_path.display()
        );
    }

    let work_base = std::env::temp_dir().join(format!("lkh_subset_bench_{}", std::process::id()));
    let _ = fs::remove_dir_all(&work_base);

    let mut classic_all: Vec<ClassicRow> = Vec::new();
    let mut lkh_all: Vec<LkhRow> = Vec::new();
    let mut task_n: HashMap<String, usize> = HashMap::new();

    for (ti, (task_file, optimal)) in SUBSET_TASKS.iter().enumerate() {
        let path = matrix_dir.join(task_file);
        eprintln!("=== {} (opt {}) ===", task_file, optimal);
        if !path.exists() {
            eprintln!("  skip: нет файла {:?}", path);
            continue;
        }

        let dm = match load_euc_2d(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  skip load: {e}");
                continue;
            }
        };
        let n = dm.n();
        task_n.insert((*task_file).to_string(), n);
        let p = adaptive_parameters(n);

        for run in 0..num_runs {
            let seed = args.seed_base.wrapping_add(ti as u64 * 10_000).wrapping_add(run as u64);
            let mut algo = AbcClassic::new(
                dm.clone(),
                p.num_employed_bees,
                p.num_onlooker_bees,
                p.limit,
                p.patience,
            );
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let (_tour, dist) = algo.run(&mut rng, p.max_iterations, false, 10_000);
            let time_s = algo.last_run_elapsed_secs().unwrap_or(0.0);
            let gap_pct = if *optimal > 0.0 {
                (dist - optimal) / optimal * 100.0
            } else {
                0.0
            };
            let success = gap_pct.abs() <= SUCCESS_TOLERANCE_PCT;
            eprintln!(
                "  classic run {}/{} | {:.2}s | dist {:.2} | gap {:+.3}% | {}",
                run + 1,
                num_runs,
                time_s,
                dist,
                gap_pct,
                if success { "OK" } else { "FAIL" }
            );
            classic_all.push(ClassicRow {
                task: (*task_file).to_string(),
                run: run + 1,
                time_s,
                distance: dist,
                optimal: *optimal,
                gap_pct,
                success,
            });
        }

        for run in 0..num_runs {
            if !lkh_ok {
                lkh_all.push(LkhRow {
                    task: (*task_file).to_string(),
                    run: run + 1,
                    time_s: None,
                    distance: None,
                    optimal: *optimal,
                    gap_pct: None,
                    success: false,
                    status: "SKIP",
                    note: "LKH binary not found".to_string(),
                });
                continue;
            }
            let run_dir = work_base.join(format!(
                "{}_{}",
                task_file.trim_end_matches(".tsp"),
                run
            ));
            let _ = fs::remove_dir_all(&run_dir);
            fs::create_dir_all(&run_dir)?;
            let seed = args
                .seed_base
                .wrapping_add(1_000_000)
                .wrapping_add(ti as u64 * 10_000)
                .wrapping_add(run as u64);

            match lkh_runner.run(&path, &run_dir, 100_000, 120, seed) {
                Ok(r) => {
                    let dist = r.cost;
                    let gap_pct = if *optimal > 0.0 {
                        (dist - optimal) / optimal * 100.0
                    } else {
                        0.0
                    };
                    let success = gap_pct.abs() <= SUCCESS_TOLERANCE_PCT;
                    eprintln!(
                        "  LKH run {}/{} | {:.2}s | dist {:.2} | gap {:+.3}% | {}",
                        run + 1,
                        num_runs,
                        r.time_sec,
                        dist,
                        gap_pct,
                        if success { "OK" } else { "FAIL" }
                    );
                    lkh_all.push(LkhRow {
                        task: (*task_file).to_string(),
                        run: run + 1,
                        time_s: Some(r.time_sec),
                        distance: Some(dist),
                        optimal: *optimal,
                        gap_pct: Some(gap_pct),
                        success,
                        status: "OK",
                        note: String::new(),
                    });
                }
                Err(e) => {
                    eprintln!("  LKH run {}/{} | ERROR: {e}", run + 1, num_runs);
                    lkh_all.push(LkhRow {
                        task: (*task_file).to_string(),
                        run: run + 1,
                        time_s: None,
                        distance: None,
                        optimal: *optimal,
                        gap_pct: None,
                        success: false,
                        status: "ERROR",
                        note: e.to_string(),
                    });
                }
            }
            let _ = fs::remove_dir_all(&run_dir);
        }
    }

    let _ = fs::remove_dir_all(&work_base);

    write_classic_csv(&args.output_classic_csv, &classic_all, &task_n)?;
    write_lkh_csv(&args.output_lkh_csv, &lkh_all, &task_n)?;
    eprintln!(
        "\nЗаписано:\n  {}\n  {}",
        args.output_classic_csv.display(),
        args.output_lkh_csv.display()
    );
    Ok(())
}

fn write_classic_csv(
    path: &Path,
    rows: &[ClassicRow],
    task_n: &HashMap<String, usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_writer(File::create(path)?);
    wtr.write_record([
        "task",
        "run",
        "time_s",
        "distance",
        "optimal",
        "gap_pct",
        "success",
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
        ])?;
    }
    wtr.flush()?;
    drop(wtr);

    append_classic_summary(path, rows, task_n)?;
    Ok(())
}

fn append_classic_summary(
    path: &Path,
    rows: &[ClassicRow],
    task_n: &HashMap<String, usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = OpenOptions::new().append(true).open(path)?;
    writeln!(f)?;
    writeln!(
        f,
        "Задача,N,Оптимум,\"Ср. время, с\",\"Ср. решение\",\"Отклонение, %\",\"Успешность, %\""
    )?;

    for (task_file, optimal) in SUBSET_TASKS {
        let task = *task_file;
        let subset: Vec<&ClassicRow> = rows.iter().filter(|r| r.task == task).collect();
        if subset.is_empty() {
            continue;
        }
        let n = task_n.get(task).copied().unwrap_or(0);
        let mean_t = subset.iter().map(|r| r.time_s).sum::<f64>() / subset.len() as f64;
        let mean_d = subset.iter().map(|r| r.distance).sum::<f64>() / subset.len() as f64;
        let dev_pct = if *optimal > 0.0 {
            (mean_d - optimal) / optimal * 100.0
        } else {
            0.0
        };
        let denom = subset.len().max(1) as f64;
        let succ_pct = 100.0 * subset.iter().filter(|r| r.success).count() as f64 / denom;
        writeln!(
            f,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.2}",
            task, n, optimal, mean_t, mean_d, dev_pct, succ_pct
        )?;
    }
    Ok(())
}

fn write_lkh_csv(
    path: &Path,
    rows: &[LkhRow],
    task_n: &HashMap<String, usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_writer(File::create(path)?);
    wtr.write_record([
        "task",
        "run",
        "time_s",
        "distance",
        "optimal",
        "gap_pct",
        "success",
        "status",
        "note",
    ])?;
    for r in rows {
        wtr.write_record([
            &r.task,
            &r.run.to_string(),
            &r.time_s.map(|x| format!("{:.6}", x)).unwrap_or_default(),
            &r.distance.map(|x| format!("{:.6}", x)).unwrap_or_default(),
            &format!("{:.6}", r.optimal),
            &r.gap_pct.map(|x| format!("{:.6}", x)).unwrap_or_default(),
            &r.success.to_string(),
            r.status,
            &r.note,
        ])?;
    }
    wtr.flush()?;
    drop(wtr);

    append_lkh_summary(path, rows, task_n)?;
    Ok(())
}

fn append_lkh_summary(
    path: &Path,
    rows: &[LkhRow],
    task_n: &HashMap<String, usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = OpenOptions::new().append(true).open(path)?;
    writeln!(f)?;
    writeln!(
        f,
        "Задача,N,Оптимум,\"Ср. время, с\",\"Ср. решение\",\"Отклонение, %\",\"Успешность, %\""
    )?;

    for (task_file, optimal) in SUBSET_TASKS {
        let task = *task_file;
        let subset: Vec<&LkhRow> = rows.iter().filter(|r| r.task == task).collect();
        if subset.is_empty() {
            continue;
        }
        let n = task_n.get(task).copied().unwrap_or(0);
        let ok: Vec<&LkhRow> = subset
            .iter()
            .copied()
            .filter(|r| r.status == "OK" && r.distance.is_some())
            .collect();

        if ok.is_empty() {
            writeln!(
                f,
                "{},{},{:.6},,,,0.00",
                task, n, optimal
            )?;
            continue;
        }

        let mean_t = ok.iter().filter_map(|r| r.time_s).sum::<f64>() / ok.len() as f64;
        let mean_d = ok.iter().filter_map(|r| r.distance).sum::<f64>() / ok.len() as f64;
        let dev_pct = if *optimal > 0.0 {
            (mean_d - optimal) / optimal * 100.0
        } else {
            0.0
        };
        let denom = subset.len().max(1) as f64;
        let succ_pct = 100.0 * subset.iter().filter(|r| r.success).count() as f64 / denom;
        writeln!(
            f,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.2}",
            task, n, optimal, mean_t, mean_d, dev_pct, succ_pct
        )?;
    }
    Ok(())
}
