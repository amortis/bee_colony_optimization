from ABC import ABCAlgorithm
from matrix_task.tsp_task import load_tsplib_instance
import numpy as np

# Загружаем матрицу расстояний из TSPLIB-задачи
# При желании можно заменить файл на другой, например: "matrix_task/eil51.tsp"
DISTANCE_MATRIX, OPTIMAL_VALUE = load_tsplib_instance("matrix_task/rd400.tsp")
print(f"Оптимальное значение: {OPTIMAL_VALUE}")

# Преобразуем в numpy array для совместимости
DISTANCE_MATRIX = np.asarray(DISTANCE_MATRIX, dtype=np.float64)

def calculate_route_distance(solution) -> float:
    """Вычисление длины маршрута."""
    total_distance = 0
    n = len(solution)
    for i in range(n):
        city1 = solution[i]
        city2 = solution[(i + 1) % n]
        total_distance += DISTANCE_MATRIX[city1, city2]
    return total_distance


# Параметры алгоритма
lb = 0  # Нумерация городов с 0
ub = len(DISTANCE_MATRIX) - 1
num_employed_bees = 300
num_onlooker_bees = 500
limit = 200  # Максимальное количество неудач для одной пчелы
max_iterations = 3500
patience = 200

# Проверки матрицы
assert len(DISTANCE_MATRIX) > 0, "Матрица пустая"
assert all(len(row) == len(DISTANCE_MATRIX) for row in DISTANCE_MATRIX), "Матрица не квадратная"


history_results = []

for _ in range(1):
    # Инициализация и запуск ABC алгоритма
    # Теперь можно передавать только distance_matrix, fitness_function создается автоматически
    abc = ABCAlgorithm(
        lb=lb, 
        ub=ub, 
        num_employed_bees=num_employed_bees, 
        num_onlooker_bees=num_onlooker_bees, 
        limit=limit, 
        patience=patience, 
        optimal_length=OPTIMAL_VALUE, 
        distance_matrix=DISTANCE_MATRIX,
        local_search_interval=10,  # Каждые 10 итераций применяем эффективный 2-opt
        elitism_rate=0.1,
        heuristic_init_ratio=0.8  # 10% решений инициализируются эвристиками (остальные случайные)
    )

    best_solution, best_fitness = abc.run_algorithm(max_iterations)
    print("\nРезультаты:")
    print("Лучший маршрут:", best_solution)
    best_distance = calculate_route_distance(best_solution)
    print("Длина маршрута:", best_distance)
    print("Фитнес:", best_fitness)
    if OPTIMAL_VALUE is not None:
        gap = best_distance - OPTIMAL_VALUE
        gap_percent = (gap / OPTIMAL_VALUE) * 100
        print("Оптимальное значение из TSPLIB:", OPTIMAL_VALUE)
        print(f"Gap: {gap:.2f} ({gap_percent:.2f}%)")

    history_results.append(best_distance)

print(history_results)
