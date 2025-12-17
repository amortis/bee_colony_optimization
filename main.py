from ABC import ABCAlgorithm
from matrix_task.tsp_task import load_tsplib_instance


# Загружаем матрицу расстояний из TSPLIB-задачи
# При желании можно заменить файл на другой, например: "matrix_task/eil51.tsp"
DISTANCE_MATRIX, OPTIMAL_VALUE = load_tsplib_instance("matrix_task/st70.tsp")
print(OPTIMAL_VALUE)

# Фитнес-функция
def fitness_function(solution):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += DISTANCE_MATRIX[solution[i]][solution[i + 1]]
    total_distance += DISTANCE_MATRIX[solution[-1]][solution[0]]
    return 1 / total_distance  # Чем больше - тем лучше


def calculate_route_distance(solution) -> int:
    """Вычисление длины маршрута."""
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += DISTANCE_MATRIX[solution[i]][solution[i + 1]]
    total_distance += DISTANCE_MATRIX[solution[-1]][solution[0]]
    return total_distance  # type: ignore


# Параметры алгоритма
lb = 0  # Нумерация городов с 0
ub = len(DISTANCE_MATRIX) - 1
num_employed_bees = 130
num_onlooker_bees = 400
limit = 100  # Максимальное количество неудач для одной пчелы
max_iterations = 3500
patience = 100

# проверки матрицы
assert len(DISTANCE_MATRIX) > 0, "Матрица пустая"
assert all(len(row) == len(DISTANCE_MATRIX) for row in DISTANCE_MATRIX), "Матрица не квадратная"


history_results = []

for _ in range(1):
    # Инициализация и запуск
    abc = ABCAlgorithm(fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, limit, patience, OPTIMAL_VALUE)

    best_solution, best_fitness = abc.run_algorithm(max_iterations)
    print("\nРезультаты:")
    print("Лучший маршрут:", best_solution)
    print("Длина маршрута:", calculate_route_distance(best_solution))
    print("Фитнес:", best_fitness)
    if OPTIMAL_VALUE is not None:
        print("Оптимальное значение из TSPLIB:", OPTIMAL_VALUE)

    history_results.append(calculate_route_distance(best_solution))

print(history_results)
