from ILS_solver import TSPSolver
from matrix_task.tsp_task import load_tsplib_instance


# Загружаем матрицу расстояний из TSPLIB-задачи
# При желании можно заменить файл на другой, например: "matrix_task/eil51.tsp"
DISTANCE_MATRIX, OPTIMAL_VALUE = load_tsplib_instance("matrix_task/ch130.tsp")
print(f"Оптимальное значение: {OPTIMAL_VALUE}")

# Проверки матрицы
assert len(DISTANCE_MATRIX) > 0, "Матрица пустая"
assert all(len(row) == len(DISTANCE_MATRIX) for row in DISTANCE_MATRIX), "Матрица не квадратная"

NUM_CITIES = len(DISTANCE_MATRIX)
MAX_TIME_SECONDS = 300  # 5 минут (можно увеличить для лучшего качества)

print(f"Количество городов: {NUM_CITIES}")
print(f"Максимальное время работы: {MAX_TIME_SECONDS} секунд")
print("\n" + "="*60)
print("Запуск ILS алгоритма (Iterated Local Search)...")
print("="*60 + "\n")

# Создаём и запускаем ILS решатель
solver = TSPSolver(
    distance_matrix=DISTANCE_MATRIX, 
    num_cities=NUM_CITIES, 
    max_time_seconds=MAX_TIME_SECONDS,
    seed=42
)

best_tour, best_distance = solver.solve()

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ:")
print("="*60)
print(f"Лучшее расстояние: {best_distance:.2f}")
if OPTIMAL_VALUE is not None:
    gap = best_distance - OPTIMAL_VALUE
    gap_percent = (gap / OPTIMAL_VALUE) * 100
    print(f"Оптимальное значение: {OPTIMAL_VALUE:.2f}")
    print(f"Gap: {gap:.2f} ({gap_percent:.2f}%)")
print(f"Время выполнения: {solver.get_formatted_time()}")
print(f"Количество итераций: {len(solver.history)}")
print(f"Лучший маршрут (первые 20 городов): {best_tour[:20]}...")
if len(best_tour) > 20:
    print(f"Лучший маршрут (последние 10 городов): ...{best_tour[-10:]}")
print("="*60)
