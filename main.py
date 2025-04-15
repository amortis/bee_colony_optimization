from ABC import ABCAlgorithm
from distane_matrix import DISTANCE_MATRIX


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
num_employed_bees = 400
num_onlooker_bees = 100
limit = 30  # Максимальное количество неудач для одной пчелы
max_iterations = 1500
patience = 75

# Инициализация и запуск
abc = ABCAlgorithm(fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, limit, patience)

#abc.employed_bee_phase()
# Результаты
# print("Employed Bee Phase -----------")
# print("Лучший маршрут:", abc.best_solution)
# print("Длина маршрута:", calculate_route_distance(abc.best_solution))
# print("Фитнес:", abc.best_fitness)

#Информация по пчелам (с trial)
#print("\nДетали по рабочим пчелам:")
# for i, bee in enumerate(abc.employed_bees):
#     print(f"Пчела {i}: Маршрут {bee.solution}, "
#           f"Длина {calculate_route_distance(bee.solution)}, "
#           f"Фитнес {bee.fitness}, "
#           f"Неудач {bee.trial}")

#abc.onlooker_bee_phase()
# Результаты
# print("\nOnLooker Bee Phase -----------")
# print("Лучший маршрут:", abc.best_solution)
# print("Длина маршрута:", calculate_route_distance(abc.best_solution))
# print("Фитнес:", abc.best_fitness)


# Информация по пчелам (с trial)
#print("\nДетали по пчелам наблюдателями:")
# for i, bee in enumerate(abc.onlooker_bees):
#     print(f"Пчела {i}: Маршрут {bee.solution}, "
#           f"Длина {calculate_route_distance(bee.solution)}, "
#           f"Фитнес {bee.fitness}, "
#           f"Неудач {bee.trial}")

#abc.visualisation()


best_solution, best_fitness = abc.run_algorithm(max_iterations)
print("\nРезультаты:")
print("Лучший маршрут:", best_solution)
print("Длина маршрута:", calculate_route_distance(best_solution))
print("Фитнес:", best_fitness)
