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
    [0, 29, 82, 46, 68, 52, 72, 42, 51, 55],  # Расстояния от города 0
    [29, 0, 55, 46, 42, 43, 43, 23, 23, 31],  # Расстояния от города 1
    [82, 55, 0, 68, 46, 55, 23, 43, 41, 29],  # Расстояния от города 2
    [46, 46, 68, 0, 82, 15, 72, 31, 62, 42],  # Расстояния от города 3
    [68, 42, 46, 82, 0, 74, 23, 52, 21, 46],  # Расстояния от города 4
    [52, 43, 55, 15, 74, 0, 61, 23, 55, 31],  # Расстояния от города 5
    [72, 43, 23, 72, 23, 61, 0, 42, 23, 31],  # Расстояния от города 6
    [42, 23, 43, 31, 52, 23, 42, 0, 33, 15],  # Расстояния от города 7
    [51, 23, 41, 62, 21, 55, 23, 33, 0, 29],  # Расстояния от города 8
    [55, 31, 29, 42, 46, 31, 31, 15, 29, 0]   # Расстояния от города 9
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
print("Employed Bee Phase -----------")
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

abc.onlooker_bee_phase()
# Результаты
print("\nOnLooker Bee Phase -----------")
print("Лучший маршрут:", abc.best_solution)
print("Длина маршрута:", calculate_route_distance(abc.best_solution))
print("Фитнес:", abc.best_fitness)

# Информация по пчелам (с trial)
print("\nДетали по рабочим пчелам:")
for i, bee in enumerate(abc.onlooker_bees):
    print(f"Пчела {i}: Маршрут {bee.solution}, "
          f"Длина {calculate_route_distance(bee.solution)}, "
          f"Фитнес {bee.fitness}, "
          f"Неудач {bee.trial}")