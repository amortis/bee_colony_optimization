from ABC import ABCAlgorithm
# Пример фитнес-функции для задачи коммивояжёра
def fitness_function(solution):
    total_distance = 0
    for i in range(len(solution) - 1):
        city_a = i
        city_b = i + 1
        total_distance += distance_matrix[city_a][city_b]
    # Добавляем расстояние от последнего города к первому (замыкаем маршрут)
    total_distance += distance_matrix[solution[-1]][solution[0]]
    return 1 / total_distance  # Фитнес = 1 / длина_маршрута

# Параметры алгоритма
lb = 0  # Нижняя граница (например, минимальный номер города)
ub = 3  # Верхняя граница (например, максимальный номер города)
num_employed_bees = 10  # Количество рабочих пчел
num_onlooker_bees = 10  # Количество пчел-наблюдателей
limit = 5  # Максимальное количество неудачных попыток

distance_matrix = [
    [0, 10, 15, 20],  # Расстояния от города 0
    [10, 0, 35, 25],   # Расстояния от города 1
    [15, 35, 0, 30],   # Расстояния от города 2
    [20, 25, 30, 0]    # Расстояния от города 3
]

# Создаем алгоритм
abc_algorithm = ABCAlgorithm(fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, limit)

# Инициализируем популяцию
abc_algorithm.initialize_population()

# Выполняем фазу занятых пчел
abc_algorithm.employed_bee_phase()

print("Лучшее решение после фазы занятых пчел:", abc_algorithm.best_solution)
print("Фитнес лучшего решения:", abc_algorithm.best_fitness)