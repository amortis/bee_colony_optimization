from ABC import ABCAlgorithm
from distane_matrix import DISTANCE_MATRIX, OPTIMAL_LENGTH



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
    return total_distance



# Параметры алгоритма
lb = 0  # Нумерация городов с 0
ub = len(DISTANCE_MATRIX) - 1
num_employed_bees = 200
num_onlooker_bees = 30
limit = 30  # Максимальное количество неудач для одной пчелы
max_iterations = 1500
patience = 300

solutions = []
for _ in range(1):
    # Инициализация и запуск
    abc = ABCAlgorithm(fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, limit, patience)


    best_solution, best_fitness = abc.run_algorithm(max_iterations)
    print("\nРезультаты:")
    print("Лучший маршрут:", best_solution)
    print("Длина маршрута:", calculate_route_distance(best_solution))
    print("Фитнес:", best_fitness)
    solutions.append(calculate_route_distance(best_solution))

print(solutions)
print([i - OPTIMAL_LENGTH for i in solutions])
print(min(solutions))
