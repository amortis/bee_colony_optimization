from ABC import ABCAlgorithm

# Фитнес-функция
def fitness_function(solution, distance_matrix):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += distance_matrix[solution[i]][solution[i + 1]]
    total_distance += distance_matrix[solution[-1]][solution[0]]
    return 1 / total_distance
# Пример матрицы расстояний для 4 городов
distance_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Параметры алгоритма
lb = 0  # Нумерация городов с 0
ub = len(distance_matrix) - 1
num_employed_bees = 10
num_onlooker_bees = 10
limit = 5

# Создаем и запускаем алгоритм
abc = ABCAlgorithm(fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, limit)
abc.initialize_population()
abc.employed_bee_phase()

# Выводим результаты
print("Лучший маршрут:", abc.best_solution)
print("Длина маршрута:", abc.calculate_route_distance(abc.best_solution))
print("Фитнес:", abc.best_fitness)

# Выводим информацию по всем пчелам
print("\nДетали по рабочим пчелам:")
for i, bee in enumerate(abc.employed_bees):
    print(f"Пчела {i}: Маршрут {bee.solution}, Длина {abc.calculate_route_distance(bee.solution)}, Фитнес {bee.fitness}, Неудач {bee.trial}")