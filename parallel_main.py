# Импортируйте вашу целевую функцию и матрицу расстояний
from parallel_abc import ABCAlgorithm  # Импортируем основной класс
from matrix_task.distane_matrix import DISTANCE_MATRIX, OPTIMAL_LENGTH # Предполагаем, что OPTIMAL_LENGTH там
import time

# Старая фитнес-функция (без изменений)
def fitness_function(solution):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += DISTANCE_MATRIX[solution[i]][solution[i + 1]]
    total_distance += DISTANCE_MATRIX[solution[-1]][solution[0]]
    return 1 / (total_distance + 1e-9) # Добавим небольшое значение, чтобы избежать деления на 0

def calculate_route_distance(solution):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += DISTANCE_MATRIX[solution[i]][solution[i + 1]]
    total_distance += DISTANCE_MATRIX[solution[-1]][solution[0]]
    return total_distance

# Параметры
lb = 0
ub = len(DISTANCE_MATRIX) - 1
num_employed_bees = 200
num_onlooker_bees = 300
limit = 100
max_iterations = 1000
patience = 200
max_workers = 4 # Укажите количество потоков

# Запускаем параллельную версию
print("🚀 Запускаем ускоренную версию...")
start_time = time.time()

# Используем ABCAlgorithm с многопоточностью
abc = ABCAlgorithm(
    fitness_function=fitness_function,
    lb=lb,
    ub=ub,
    num_employed_bees=num_employed_bees,
    num_onlooker_bees=num_onlooker_bees,
    limit=limit,
    patience=patience,
    max_workers=max_workers # Передаем количество потоков
)
best_solution, best_fitness = abc.run_algorithm(max_iterations) # Используем run_algorithm

total_time = time.time() - start_time
distance = calculate_route_distance(best_solution)

print(f"\n✅ Результаты:")
print(f"Время: {total_time:.2f} сек")
print(f"Длина маршрута: {distance}")
print(f"Фитнес: {best_fitness}")
print(f"Маршрут: {best_solution}")
