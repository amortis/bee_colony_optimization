//! Interactive console like repo root `main.py`: pick task, adaptive params, run ABCTSPILS.

use abc_rust::adaptive::adaptive_parameters;
use abc_rust::load_euc_2d;
use abc_rust::task_sets::MENU_TASKS;
use abc_rust::{AbcConfig, AbcTspIls};
use clap::Parser;
use rand::thread_rng;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Interactive TSP menu (same flow as main.py)")]
struct Args {
    /// Directory with `.tsp` files (like `matrix_task/` in the Python project)
    #[arg(long, default_value = "../matrix_task")]
    matrix_dir: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    print_menu();
    let idx = read_choice(MENU_TASKS.len())?;
    let (task_name, optimal) = MENU_TASKS[idx];
    let path = args.matrix_dir.join(task_name);

    if !path.exists() {
        eprintln!("Файл не найден: {}", path.display());
        eprintln!("Укажите каталог с задачами: --matrix-dir");
        return Ok(());
    }

    let dm = load_euc_2d(&path)?;
    let num_cities = dm.n();
    println!("TSPLIB instance loaded.");
    println!("Known optimal value from TSPLIB (or reference): {optimal}");

    let p = adaptive_parameters(num_cities);
    println!("\nПараметры для задачи с {num_cities} городами:");
    println!("{:<28} {}", "Параметр", "Значение");
    println!("{:-<40}", "");
    println!("{:<28} {}", "num_employed_bees", p.num_employed_bees);
    println!("{:<28} {}", "num_onlooker_bees", p.num_onlooker_bees);
    println!("{:<28} {}", "limit", p.limit);
    println!("{:<28} {}", "patience", p.patience);
    println!("{:<28} {}", "local_search_interval", p.local_search_interval);
    println!("{:<28} {:.2}", "heuristic_init_ratio", p.heuristic_init_ratio);
    println!("{:<28} {}", "max_iterations", p.max_iterations);
    println!();

    let config = AbcConfig {
        num_employed_bees: p.num_employed_bees,
        num_onlooker_bees: p.num_onlooker_bees,
        limit: p.limit,
        patience: p.patience,
        local_search_interval: p.local_search_interval,
        heuristic_init_ratio: p.heuristic_init_ratio,
        optimal_known: Some(optimal),
        use_parallel: false,
        num_workers: Some(16),
    };

    let mut abc_ils = AbcTspIls::new(dm, config);
    let mut rng = thread_rng();
    let (best_tour, best_distance) = abc_ils.run(&mut rng, p.max_iterations);

    println!("\n--- FINAL RESULT ---");
    println!("Best tour: {best_tour:?}");
    println!("Best distance: {best_distance:.2}");
    if let Some(sec) = abc_ils.last_run_elapsed_secs() {
        println!("Wall time (search): {sec:.3} s");
    }
    if optimal > 0.0 {
        let gap = best_distance - optimal;
        let gap_percent = gap / optimal * 100.0;
        println!("Gap to optimal: {gap:.2} ({gap_percent:.2}%)");
    }

    Ok(())
}

fn print_menu() {
    println!("\nДоступные задачи TSP:");
    println!("{:<4} {:<16} {:>18}", "№", "Файл", "Оптимальное значение");
    println!("{:-<42}", "");
    for (i, (name, opt)) in MENU_TASKS.iter().enumerate() {
        println!("{:<4} {:<16} {:>18.0}", i + 1, name, opt);
    }
    println!();
}

fn read_choice(max: usize) -> Result<usize, io::Error> {
    loop {
        print!("Выберите номер задачи (1-{max}): ");
        io::stdout().flush()?;
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        let t = line.trim();
        if let Ok(n) = t.parse::<usize>() {
            if (1..=max).contains(&n) {
                return Ok(n - 1);
            }
        }
        println!("\nНеправильный номер задачи!");
        print_menu();
    }
}
