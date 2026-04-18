#!/usr/bin/env python3
"""
Пакетное тестирование классического алгоритма ABC (ABC_Classic)
на наборе задач TSPLIB.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import csv
import numpy as np
from typing import List, Dict
from ABC_classic import ABC_Classic
from matrix_task.tsp_task import TspTask

# ================= НАСТРОЙКИ =================
NUM_RUNS = 1
SUCCESS_TOLERANCE = 0.1  # Допустимое отклонение от оптимума в %
OUTPUT_CSV = "benchmark_classic.csv"

TASKS = [
    ("eil51.tsp", 426),
    ("st70.tsp", 675),
    ("eil101.tsp", 629),
    ("rd100.tsp", 7910),
    ("a280.tsp", 2579),
]

def get_adaptive_parameters_classic(num_cities: int) -> Dict:
    """Адаптивные параметры для классического ABC."""
    if num_cities < 100:
        return {'num_employed_bees': 100, 'num_onlooker_bees': 60, 'limit': 50, 'patience': 300, 'max_iterations': 2000}
    elif num_cities < 200:
        return {'num_employed_bees': 150, 'num_onlooker_bees': 80, 'limit': 75, 'patience': 400, 'max_iterations': 3000}
    elif num_cities < 300:
        return {'num_employed_bees': 200, 'num_onlooker_bees': 120, 'limit': 100, 'patience': 500, 'max_iterations': 4000}
    elif num_cities < 500:
        return {'num_employed_bees': 300, 'num_onlooker_bees': 150, 'limit': 120, 'patience': 600, 'max_iterations': 5000}
    else:
        return {'num_employed_bees': max(400, num_cities // 2), 'num_onlooker_bees': max(200, num_cities // 3), 
                'limit': 150, 'patience': 700, 'max_iterations': 6000}

def run_single_task(task_file: str, optimal: int) -> List[Dict]:
    results = []
    try:
        task = TspTask(task_file, optimal)
    except Exception as e:
        print(f"⚠️ Ошибка загрузки {task_file}: {e}. Пропуск.")
        return results

    num_cities = len(task.distance_matrix)
    base_params = get_adaptive_parameters_classic(num_cities)
    print(f"\n📦 ЗАПУСК: {task_file} (N={num_cities}, Opt: {optimal})")

    for run_idx in range(NUM_RUNS):
        print(f"  🔄 Запуск {run_idx+1}/{NUM_RUNS}...", end=" ", flush=True)
        
        current_params = base_params.copy()
        current_params['seed'] = run_idx + 42
        max_iter = current_params.pop('max_iterations')
        
        start = time.time()
        
        # ✅ Инициализация через distance_matrix
        abc = ABC_Classic(
            distance_matrix=task.distance_matrix,
            lb=0,
            ub=num_cities - 1,
            **current_params
        )
        
        # ✅ run() возвращает (tour, distance)
        best_tour, best_distance = abc.run(max_iterations=max_iter, verbose=False)
        elapsed = time.time() - start

        gap_pct = ((best_distance - optimal) / optimal * 100) if optimal > 0 else 0
        success = abs(gap_pct) <= SUCCESS_TOLERANCE
        print(f"⏱️ {elapsed:.1f}s | 📏 {best_distance:.2f} | 📉 {gap_pct:+.3f}% | ✅ {'OK' if success else 'FAIL'}")
        
        results.append({
            'task': task_file, 'n_cities': num_cities, 'run': run_idx + 1,
            'optimal': optimal, 'found_distance': best_distance,
            'gap_abs': best_distance - optimal, 'gap_pct': gap_pct,
            'time_sec': elapsed, 'success': success,
            'iterations': len(abc.get_convergence_history())
        })

    # Агрегация
    times = [r['time_sec'] for r in results]
    distances = [r['found_distance'] for r in results]
    gaps = [r['gap_pct'] for r in results]
    success_rate = sum(r['success'] for r in results) / NUM_RUNS * 100
    
    print(f"  📊 ИТОГО: Время={np.mean(times):.1f}±{np.std(times):.1f}s | "
          f"Решение={np.mean(distances):.2f}±{np.std(distances):.2f} | "
          f"Gap={np.mean(gaps):+.3f}%±{np.std(gaps):.3f}% | "
          f"Успешность={success_rate:.0f}%\n")
    return results

def main():
    print("="*70)
    print("🐝 ПАКЕТНОЕ ТЕСТИРОВАНИЕ: Классический ABC")
    print("="*70)
    
    all_results: List[Dict] = []
    for task_file, optimal in TASKS:
        all_results.extend(run_single_task(task_file, optimal))
    
    if all_results:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['task', 'n_cities', 'run', 'optimal', 'found_distance', 
                          'gap_abs', 'gap_pct', 'time_sec', 'success', 'iterations']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"✅ Результаты сохранены в {OUTPUT_CSV}")
    
    print("\n" + "="*70)
    print("📋 СВОДНАЯ ТАБЛИЦА")
    print("="*70)
    print(f"{'Задача':<14} | {'N':>4} | {'Оптимум':>8} | {'Время(с)':>12} | {'Gap%':>10} | {'Успешность':>10}")
    print("-"*70)
    
    for task_file, optimal in TASKS:
        task_data = [r for r in all_results if r['task'] == task_file]
        if not task_data: continue
        times = [r['time_sec'] for r in task_data]
        gaps = [r['gap_pct'] for r in task_data]
        success_rate = sum(r['success'] for r in task_data) / len(task_data) * 100
        print(f"{task_file:<14} | {task_data[0]['n_cities']:>4} | {optimal:>8} | "
              f"{np.mean(times):>8.1f}±{np.std(times):.1f} | "
              f"{np.mean(gaps):>+.3f}±{np.std(gaps):.3f}% | {success_rate:>9.0f}%")
    print("="*70)

if __name__ == "__main__":
    main()