#!/usr/bin/env python3
"""
Скрипт пакетного запуска LKH на наборе TSPLIB-задач.
Выводит: найденное решение, известный оптимум, разницу (абс. и %), время, статус.
Ожидается, что файл лежит в папке LKH/, а бинарник в LKH/LKH-2.0.7/LKH
Задачи лежат в ../matrix_task/
"""

import os
import subprocess
import re
import csv
import time
from pathlib import Path

# ================= НАСТРОЙКИ ПУТЕЙ =================
SCRIPT_DIR = Path(__file__).resolve().parent
LKH_BIN = SCRIPT_DIR / "LKH-2.0.7" / "LKH"
TSP_DIR = SCRIPT_DIR.parent / "matrix_task"
OUTPUT_CSV = SCRIPT_DIR / "lkh_results.csv"
# ===================================================

# Список задач: (имя_файла, известный_оптимум из TSPLIB)
TASKS = [
    ("eil51.tsp", 426),
    ("st70.tsp", 675),
    ("eil101.tsp", 629),
    ("rd100.tsp", 7910),
    ("a280.tsp", 2579),
    ("lin318.tsp", 42029),
    ("rat575.tsp", 6773),
    ("pa561.tsp", 2763),
    ("rd400.tsp", 15281),
    ("pr1002.tsp", 259045)
]

TIMEOUT = 600  # Лимит секунд на одну задачу
RESULTS = []

def run_lkh(tsp_file: str, known_optimal: int) -> dict:
    tsp_path = TSP_DIR / tsp_file
    if not tsp_path.exists():
        print(f"⚠️ Файл {tsp_path} не найден. Пропуск.")
        return {
            "task": tsp_file, "known_optimal": known_optimal, "found_solution": None,
            "abs_diff": None, "gap_pct": None, "time_sec": None, "status": "FILE_NOT_FOUND"
        }

    abs_tsp = tsp_path.resolve()
    abs_tour = (SCRIPT_DIR / f"{tsp_file}.tour").resolve()
    par_file = SCRIPT_DIR / f"{tsp_file}.par"

    # Уменьшаем MAX_TRIALS для быстрого бенчмарка. LKH находит eil51/a280 за <0.1с при 100k trials.
    par_content = f"""PROBLEM_FILE = {abs_tsp}
RUNS = 1
SEED = 42
MAX_TRIALS = 100000
TIME_LIMIT = 120
OUTPUT_TOUR_FILE = {abs_tour}
"""
    par_file.write_text(par_content)

    print(f"🔍 Запуск LKH для {tsp_file} (Opt: {known_optimal})...")
    start = time.time()
    
    try:
        res = subprocess.run(
            [str(LKH_BIN), str(par_file)],
            capture_output=True,
            text=True,
            timeout=150  # 2.5 минуты максимум
        )
        elapsed = time.time() - start
        output = res.stdout + res.stderr

        cost_match = re.search(r"Cost\s*=\s*([\d\.]+)", output)
        found_solution = float(cost_match.group(1)) if cost_match else None

        abs_diff = (found_solution - known_optimal) if found_solution is not None else None
        gap_pct = (abs_diff / known_optimal * 100) if abs_diff is not None and known_optimal > 0 else None

        # Безопасное форматирование
        diff_str = f"+{abs_diff:.2f}" if abs_diff is not None else "N/A"
        gap_str = f"+{gap_pct:.3f}" if gap_pct is not None else "N/A"
        print(f"  ✅ Found={found_solution} | Opt={known_optimal} | Δ={diff_str} | Gap={gap_str}% | Time={elapsed:.2f}s")
        
        return {
            "task": tsp_file, "known_optimal": known_optimal, "found_solution": found_solution,
            "abs_diff": abs_diff, "gap_pct": gap_pct, "time_sec": elapsed,
            "status": "OK" if found_solution is not None else "PARSE_ERROR"
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  ⏱️ Таймаут ({elapsed:.1f} с)")
        return {
            "task": tsp_file, "known_optimal": known_optimal, "found_solution": None,
            "abs_diff": None, "gap_pct": None, "time_sec": elapsed, "status": "TIMEOUT"
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ Ошибка: {e} (прошло {elapsed:.1f} с)")
        return {
            "task": tsp_file, "known_optimal": known_optimal, "found_solution": None,
            "abs_diff": None, "gap_pct": None, "time_sec": elapsed, "status": f"ERROR: {e}"
        }

def main():
    if not LKH_BIN.exists():
        print(f"❌ Бинарник LKH не найден по пути: {LKH_BIN}")
        print("Убедитесь, что он скомпилирован и имеет права на выполнение: chmod +x LKH-2.0.7/LKH")
        return

    print(f"📁 Бинарник: {LKH_BIN}")
    print(f"📁 Задачи: {TSP_DIR}")
    print("="*90)

    for tsp_file, known_opt in TASKS:
        RESULTS.append(run_lkh(tsp_file, known_opt))

    # Сохранение в CSV с нужными колонками
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "task", "known_optimal", "found_solution", 
            "abs_diff", "gap_pct", "time_sec", "status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(RESULTS)

    print(f"\n✅ Результаты сохранены в {OUTPUT_CSV}")
    
    # Красивая сводка в консоль
    print("\n📊 Сводка результатов:")
    print(f"{'Задача':<14} | {'Opt':>8} | {'Found':>8} | {'Δ (abs)':>9} | {'Gap%':>7} | {'Time(s)':>8} | Статус")
    print("-"*90)
    for r in RESULTS:
        found_str = f"{r['found_solution']:.2f}" if r['found_solution'] else "N/A"
        diff_str = f"{r['abs_diff']:+.2f}" if r['abs_diff'] is not None else "N/A"
        gap_str = f"{r['gap_pct']:+.3f}" if r['gap_pct'] is not None else "N/A"
        time_str = f"{r['time_sec']:.2f}" if r['time_sec'] else "N/A"
        print(f"{r['task']:<14} | {r['known_optimal']:>8} | {found_str:>8} | "
              f"{diff_str:>9} | {gap_str:>7} | {time_str:>8} | {r['status']}")

if __name__ == "__main__":
    main()