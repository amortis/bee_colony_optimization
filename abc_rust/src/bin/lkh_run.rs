//! Run the LKH binary on a `.tsp` file (wrapper, same idea as `LKH/run_lkh.py`).

use abc_rust::lkh::{LkhError, LkhRunner};
use clap::Parser;
use std::env;
use std::path::{Path, PathBuf};
use std::process;

#[derive(Parser, Debug)]
#[command(
    name = "lkh-run",
    version,
    about = "Запуск солвера LKH по файлу TSPLIB (.tsp)",
    long_about = "Нужен скомпилированный бинарник LKH: в репозитории это LKH/LKH-2.0.7/LKH\n\
(см. README в LKH или: cd LKH/LKH-2.0.7/SRC && make).\n\n\
Без флагов укажите --tsp или используйте --example для быстрого теста на eil51."
)]
struct Args {
    /// Путь к файлу задачи `.tsp`
    #[arg(short, long)]
    tsp: Option<PathBuf>,

    /// Быстрый тест: `matrix_task/eil51.tsp` от корня репозитория (ищется вверх от текущей папки)
    #[arg(long)]
    example: bool,

    /// Только показать, какой бинарник LKH и какой `.tsp` будут использованы (без запуска)
    #[arg(long)]
    print_paths: bool,

    /// Путь к исполняемому файлу `LKH` (по умолчанию: .../LKH/LKH-2.0.7/LKH от корня репо)
    #[arg(long)]
    lkh_bin: Option<PathBuf>,

    /// Каталог для `.par` и `.tour` (по умолчанию: временный)
    #[arg(long)]
    work_dir: Option<PathBuf>,

    #[arg(long, default_value_t = 100_000_u64)]
    max_trials: u64,

    #[arg(long, default_value_t = 120_u64)]
    time_limit: u64,

    #[arg(long, default_value_t = 42_u64)]
    seed: u64,

    /// Печатать объединённый stdout+stderr LKH (удобно, если не парсится Cost)
    #[arg(short, long)]
    verbose: bool,
}

/// Поднимаемся от cwd вверх, пока не найдём каталог с `matrix_task/` и `LKH/`.
fn find_repo_root() -> Option<PathBuf> {
    let mut p = env::current_dir().ok()?;
    for _ in 0..12 {
        if p.join("matrix_task").is_dir() {
            return Some(p);
        }
        if !p.pop() {
            break;
        }
    }
    None
}

fn default_lkh_binary() -> PathBuf {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut p = cwd.clone();
    for _ in 0..10 {
        let candidate = p.join("LKH").join("LKH-2.0.7").join("LKH");
        if candidate.exists() {
            return candidate;
        }
        if !p.pop() {
            break;
        }
    }
    cwd.join("LKH").join("LKH-2.0.7").join("LKH")
}

fn resolve_tsp(args: &Args) -> Result<PathBuf, String> {
    if args.example {
        let root = find_repo_root()
            .ok_or_else(|| "Не найден корень репозитория (нужны папки matrix_task/ и LKH/). Запустите из bee_colony_optimization/ или abc_rust/ и используйте --example, либо укажите --tsp вручную.".to_string())?;
        let p = root.join("matrix_task").join("eil51.tsp");
        if !p.exists() {
            return Err(format!("Файл не найден: {}", p.display()));
        }
        return Ok(p);
    }
    args.tsp
        .clone()
        .ok_or_else(|| "Укажите --tsp ПУТЬ/к/файлу.tsp или запустите с --example (eil51)".to_string())
}

fn main() {
    let args = Args::parse();

    let lkh_bin = args.lkh_bin.clone().unwrap_or_else(default_lkh_binary);
    let runner = LkhRunner::new(&lkh_bin);

    if args.print_paths {
        println!("LKH binary (exists={}): {}", runner.binary_exists(), lkh_bin.display());
        match resolve_tsp(&args) {
            Ok(ref p) => println!("TSP file (exists={}): {}", p.exists(), p.display()),
            Err(_) => {
                println!("TSP file: (не задан — добавьте --example или --tsp для проверки)");
            }
        }
        if !runner.binary_exists() {
            eprintln!("\nСоберите LKH: cd LKH/LKH-2.0.7/SRC && make");
            eprintln!("Затем: chmod +x {}", lkh_bin.display());
        }
        return;
    }

    let tsp_path = match resolve_tsp(&args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Ошибка: {e}");
            eprintln!();
            eprintln!("Примеры:");
            eprintln!("  lkh-run --example");
            eprintln!("  lkh-run --tsp ../matrix_task/eil51.tsp");
            eprintln!("  lkh-run --print-paths");
            process::exit(1);
        }
    };

    if !runner.binary_exists() {
        eprintln!("Бинарник LKH не найден: {}", lkh_bin.display());
        eprintln!("Соберите: cd <repo>/LKH/LKH-2.0.7/SRC && make");
        eprintln!("Или укажите путь: lkh-run --lkh-bin /путь/к/LKH --tsp ...");
        process::exit(1);
    }

    let work_dir = args.work_dir.unwrap_or_else(|| {
        env::temp_dir().join(format!("lkh_run_{}", process::id()))
    });

    eprintln!("LKH: {}", lkh_bin.display());
    eprintln!("Задача: {}", tsp_path.display());

    match runner.run(
        Path::new(&tsp_path),
        &work_dir,
        args.max_trials,
        args.time_limit,
        args.seed,
    ) {
        Ok(r) => {
            println!("cost\t{:.6}", r.cost);
            println!("time_sec\t{:.6}", r.time_sec);
            if args.verbose {
                println!("--- LKH output ---");
                println!("{}", r.stdout_stderr);
            }
        }
        Err(e) => {
            eprintln!("Ошибка: {e}");
            if let LkhError::NonZeroExit(code) = &e {
                eprintln!("(код выхода {code})");
            }
            eprintln!("\nПодсказка: lkh-run --verbose --tsp {}  (покажет вывод LKH)", tsp_path.display());
            process::exit(1);
        }
    }
}
