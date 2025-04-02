from ABC import ABCAlgorithm

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

# Матрица расстояний
DISTANCE_MATRIX = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Параметры алгоритма
lb = 0  # Нумерация городов с 0
ub = len(DISTANCE_MATRIX) - 1
num_employed_bees = 10
num_onlooker_bees = 10
limit = 5  # Максимальное количество неудач для одной пчелы

# Инициализация и запуск
abc = ABCAlgorithm(fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, limit)
abc.initialize_population()
abc.employed_bee_phase()

# Результаты
print("Лучший маршрут:", abc.best_solution)
print("Длина маршрута:", calculate_route_distance(abc.best_solution))
print("Фитнес:", abc.best_fitness)

# Информация по пчелам (с trial)
print("\nДетали по рабочим пчелам:")
for i, bee in enumerate(abc.employed_bees):
    print(f"Пчела {i}: Маршрут {bee.solution}, "
          f"Длина {calculate_route_distance(bee.solution)}, "
          f"Фитнес {bee.fitness}, "
          f"Неудач {bee.trial}")