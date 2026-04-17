#!/usr/bin/env python3
"""
Скрипт пакетного запуска LKH на наборе TSPLIB-задач.
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

# Список задач: (имя_файла, известный_оптимум)
# Примечание: для rd400.tsp стандартный оптимум TSPLIB = 15281, в вашем списке было 7190 (вероятно, копия rd100). Исправлено.
TASKS = [
    ("eil51.tsp", 426),
    ("st70.tsp", 675),
    ("eil101.tsp", 629),
    ("rd100.tsp", 7190),
    ("a280.tsp", 2579),
    ("lin318.tsp", 42029),
    ("rat575.tsp", 6773),
    ("pa561.tsp", 2763),
    ("rd400.tsp", 15281),
    ("pr1002.tsp", 259045)
]

TIMEOUT = 600  # Лимит секунд на одну задачу (LKH обычно решает их за секунды/минуты)
RESULTS = []

def run_lkh(tsp_file: str, optimal: int) -> dict:
    tsp_path = TSP_DIR / tsp_file
    if not tsp_path.exists():
        print(f"⚠️ Файл {tsp_path} не найден. Пропуск.")
        return {"task": tsp_file, "optimal": optimal, "distance": None, "time_sec": None, "gap_pct": None, "status": "FILE_NOT_FOUND"}

    # Используем абсолютные пути в .par файле для избежания проблем с CWD
    abs_tsp = tsp_path.resolve()
    abs_tour = (SCRIPT_DIR / f"{tsp_file}.tour").resolve()
    par_file = SCRIPT_DIR / f"{tsp_file}.par"

    par_content = f"""PROBLEM_FILE = {abs_tsp}
    RUNS = 1
    SEED = 42
    MAX_TRIALS = 2000000
    TIME_LIMIT = {TIMEOUT}
    OUTPUT_TOUR_FILE = {abs_tour}
    """
    par_file.write_text(par_content)

    print(f"🔍 Запуск LKH для {tsp_file} (Opt: {optimal})...")
    start = time.time()
    
    try:
        res = subprocess.run(
            [str(LKH_BIN), str(par_file)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT + 15
        )
        elapsed = time.time() - start
        output = res.stdout + res.stderr

        # Парсинг вывода LKH 2.0.7
        cost_match = re.search(r"Cost\s*=\s*([\d\.]+)", output)
        time_match = re.search(r"Time\s*=\s*([\d\.]+)\s*sec", output)

        distance = float(cost_match.group(1)) if cost_match else None
        lkh_time = float(time_match.group(1)) if time_match else elapsed

        gap_pct = ((distance - optimal) / optimal * 100) if distance and optimal > 0 else None
        status = "OK" if distance is not None else "PARSE_ERROR"

        print(f"  ✅ Cost={distance} | Time={lkh_time:.2f}s | Gap={gap_pct:+.3f}%")
        return {"task": tsp_file, "optimal": optimal, "distance": distance, "time_sec": lkh_time, "gap_pct": gap_pct, "status": status}

    except subprocess.TimeoutExpired:
        print(f"  ⏱️ Таймаут ({TIMEOUT} сек)")
        return {"task": tsp_file, "optimal": optimal, "distance": None, "time_sec": TIMEOUT, "gap_pct": None, "status": "TIMEOUT"}
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return {"task": tsp_file, "optimal": optimal, "distance": None, "time_sec": time.time()-start, "gap_pct": None, "status": f"ERROR: {e}"}

def main():
    if not LKH_BIN.exists():
        print(f"❌ Бинарник LKH не найден по пути: {LKH_BIN}")
        print("Убедитесь, что он скомпилирован и имеет права на выполнение: chmod +x LKH-2.0.7/LKH")
        return

    print(f"📁 Бинарник: {LKH_BIN}")
    print(f"📁 Задачи: {TSP_DIR}")
    print("="*70)

    for tsp_file, optimal in TASKS:
        RESULTS.append(run_lkh(tsp_file, optimal))

    # Сохранение в CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "optimal", "distance", "time_sec", "gap_pct", "status"])
        writer.writeheader()
        writer.writerows(RESULTS)

    print(f"\n✅ Результаты сохранены в {OUTPUT_CSV}")
    print("\n📊 Сводка:")
    print(f"{'Задача':<12} | {'Opt':>7} | {'LKH':>8} | {'Время(с)':>8} | {'Gap%':>6} | Статус")
    print("-" * 70)
    for r in RESULTS:
        print(f"{r['task']:<12} | {r['optimal']:>7} | {str(r['distance']):>8} | {str(r['time_sec']):>8} | {str(r['gap_pct']):>6} | {r['status']}")

if __name__ == "__main__":
    main()