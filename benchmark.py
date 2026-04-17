import time
import csv
import threading
import numpy as np
import subprocess
import os
from ABC import ABCTSPILS
from matrix_task.tsp_task import TspTask

# ================= НАСТРОЙКИ =================
NUM_RUNS = 10
SUCCESS_TOLERANCE = 0.1  # Допустимое отклонение от оптимума в %
GPU_POLL_INTERVAL = 0.5  # Частота опроса GPU (сек)

TASKS = [
    ("a280.tsp", 2579),
    ("eil51.tsp", 426),
    ("eil101.tsp", 629),
    ("lin318.tsp", 42029),
    ("pa561.tsp", 2763),
    ("pr1002.tsp", 259045),
    ("rat575.tsp", 6773),
    ("rd100.tsp", 7190),
    ("rd400.tsp", 7190),
    ("st70.tsp", 675)
]
OUTPUT_CSV = "benchmark_results.csv"
# =============================================

def get_adaptive_parameters(num_cities: int) -> dict:
    if num_cities < 100:
        return {'num_employed_bees': 15, 'num_onlooker_bees': 20, 'limit': 100, 'patience': 150, 'local_search_interval': 20, 'heuristic_init_ratio': 0.8, 'max_iterations': 1000}
    elif num_cities < 200:
        return {'num_employed_bees': 20, 'num_onlooker_bees': 25, 'limit': 100, 'patience': 200, 'local_search_interval': 25, 'heuristic_init_ratio': 0.85, 'max_iterations': 2000}
    elif num_cities < 300:
        return {'num_employed_bees': 15, 'num_onlooker_bees': 30, 'limit': 150, 'patience': 500, 'local_search_interval': 30, 'heuristic_init_ratio': 0.9, 'max_iterations': 3000}
    else:
        return {'num_employed_bees': 40, 'num_onlooker_bees': 50, 'limit': 60, 'patience': 350, 'local_search_interval': 30, 'heuristic_init_ratio': 0.95, 'max_iterations': 4000}

class GPUMonitor:
    def __init__(self, interval=GPU_POLL_INTERVAL):
        self.interval = interval
        self.recordings = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self): self._thread.start()
    def stop(self): self._stop.set(); self._thread.join(); return [x for x in self.recordings if x is not None]

    def _poll(self):
        while not self._stop.is_set():
            try:
                # Попытка через pynvml (точнее)
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.recordings.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except:
                try:
                    # Фоллбэк на nvidia-smi
                    out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"]).decode().strip()
                    self.recordings.append(float(out.replace('%', '')))
                except:
                    self.recordings.append(None)
            time.sleep(self.interval)

def run_experiment():
    all_results = []
    
    for task_file, optimal in TASKS:
        print(f"\n{'='*60}\n📦 ЗАПУСК: {task_file} (Opt: {optimal})\n{'='*60}")
        try:
            task = TspTask(task_file, optimal)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {task_file}: {e}. Пропуск.\n")
            continue
            
        num_cities = len(task.distance_matrix)
        params = get_adaptive_parameters(num_cities)
        task_runs = []

        for i in range(NUM_RUNS):
            print(f"  🔄 Запуск {i+1}/{NUM_RUNS}...", end=" ", flush=True)
            monitor = GPUMonitor()
            monitor.start()

            t0 = time.time()
            abc = ABCTSPILS(
                distance_matrix=task.distance_matrix,
                num_employed_bees=params['num_employed_bees'],
                num_onlooker_bees=params['num_onlooker_bees'],
                limit=params['limit'],
                patience=params['patience'],
                local_search_interval=params['local_search_interval'],
                heuristic_init_ratio=params['heuristic_init_ratio'],
                use_parallel=True,
                num_workers=16,
                use_gpu=True,
                optimal_value=optimal,
                visualization=False  # Отключаем графики для пакетного режима
            )
            best_tour, best_dist = abc.run(max_iterations=params['max_iterations'])
            elapsed = time.time() - t0

            avg_gpu = np.mean(monitor.stop()) if monitor.stop() else 0.0
            gap_pct = ((best_dist - optimal) / optimal * 100) if optimal > 0 else 0
            success = abs(gap_pct) <= SUCCESS_TOLERANCE

            print(f"⏱️ {elapsed:.1f}s | 📏 {best_dist:.2f} | 📉 {gap_pct:+.3f}% | 🎮 GPU: {avg_gpu:.1f}% | ✅ {'OK' if success else 'FAIL'}")
            
            task_runs.append({
                'task': task_file, 'run': i+1, 'time_s': elapsed,
                'distance': best_dist, 'optimal': optimal,
                'gap_pct': gap_pct, 'success': success, 'avg_gpu_util': avg_gpu
            })

        # Агрегация статистики по задаче
        times = [r['time_s'] for r in task_runs]
        dists = [r['distance'] for r in task_runs]
        print(f"  📊 ИТОГО: Время={np.mean(times):.1f}±{np.std(times):.1f}s | "
              f"Решение={np.mean(dists):.2f}±{np.std(dists):.2f} | "
              f"Успешность={sum(r['success'] for r in task_runs)/NUM_RUNS*100:.0f}% | "
              f"Ср. GPU={np.mean([r['avg_gpu_util'] for r in task_runs]):.1f}%\n")
        all_results.extend(task_runs)

    # Сохранение в CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'run', 'time_s', 'distance', 'optimal', 'gap_pct', 'success', 'avg_gpu_util'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"✅ Результаты сохранены в {OUTPUT_CSV}")

if __name__ == "__main__":
    run_experiment()