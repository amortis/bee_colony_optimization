import numpy as np
from ABC_sparse import ABCAlgorithmSparse
from matrix_task.sparred_tsp_task import load_tsplib_instance, create_sparse_matrix_with_hamiltonian

# Фитнес-функция для неполного графа
def fitness_function(solution):
    total_distance = 0
    n = len(solution)
    
    for i in range(n):
        city_from = solution[i]
        city_to = solution[(i + 1) % n]
        
        # Проверяем существование ребра
        if np.isinf(DISTANCE_MATRIX[city_from][city_to]):
            return 0.0  # Недопустимое решение
        
        total_distance += DISTANCE_MATRIX[city_from][city_to]
    
    # Чем меньше расстояние, тем лучше фитнес
    return 1.0 / total_distance

def calculate_route_distance(solution):
    """Вычисление длины маршрута с проверкой допустимости"""
    total_distance = 0
    n = len(solution)
    
    for i in range(n):
        city_from = solution[i]
        city_to = solution[(i + 1) % n]
        
        if np.isinf(DISTANCE_MATRIX[city_from][city_to]):
            return float('inf')  # Недопустимое решение
        
        total_distance += DISTANCE_MATRIX[city_from][city_to]
    
    return total_distance

if __name__ == "__main__":
    # Загружаем TSP задачу
    print("Загрузка TSP задачи...")
    full_matrix, optimal = load_tsplib_instance("matrix_task/st70.tsp")
    
    # Создаем неполный граф с гарантированным циклом
    print("Создание неполного графа...")
    DISTANCE_MATRIX, hamiltonian_cycle = create_sparse_matrix_with_hamiltonian(
        full_matrix, 
        keep_probability=1,  # Оставляем 30% рёбер
        seed=42
    )
    
    print(f"Городов: {len(DISTANCE_MATRIX)}")
    print(f"Гарантированный цикл: {hamiltonian_cycle[:10]}...")
    
    # Проверяем гарантированный цикл
    is_valid = True
    cycle_length = 0
    n = len(hamiltonian_cycle)
    for i in range(n):
        city_from = hamiltonian_cycle[i]
        city_to = hamiltonian_cycle[(i + 1) % n]
        if np.isinf(DISTANCE_MATRIX[city_from][city_to]):
            is_valid = False
            break
        cycle_length += DISTANCE_MATRIX[city_from][city_to]
    
    if is_valid:
        print(f"Гарантированный цикл корректен, длина: {cycle_length:.2f}")
    else:
        print("Ошибка: гарантированный цикл содержит отсутствующие ребра!")
        exit(1)
    
    task_dimension = 575

    # Параметры алгоритма
    num_employed_bees = 75
    num_onlooker_bees = 120
    limit = 200
    max_iterations = 3000
    patience = 100
    
    history_results = []
    
    # Запускаем несколько экспериментов
    for run in range(1):
        print(f"\n=== Запуск {run + 1} ===")
        
        # Инициализация и запуск
        abc = ABCAlgorithmSparse(
            fitness_function=fitness_function,
            distance_matrix=DISTANCE_MATRIX,
            num_employed_bees=num_employed_bees,
            num_onlooker_bees=num_onlooker_bees,
            limit=limit,
            patience=patience,
            num_workers=4
        )
        
        best_solution, best_fitness = abc.run_algorithm(
            max_iterations=max_iterations,
            initial_cycle=hamiltonian_cycle
        )
        
        best_distance = calculate_route_distance(best_solution)
        
        print("\nРезультаты:")
        print(f"Лучший маршрут: {best_solution}")
        print(f"Длина маршрута: {best_distance:.2f}")
        print(f"Фитнес: {best_fitness}")
        
        if np.isfinite(best_distance):
            history_results.append(best_distance)
            print(f"Улучшение относительно гарантированного цикла: {cycle_length - best_distance:.2f}")
        else:
            print("Найденное решение недопустимо!")
    
    print(f"\n=== Сводка по запускам ===")
    if history_results:
        print(f"Лучшая длина: {min(history_results):.2f}")
        print(f"Худшая длина: {max(history_results):.2f}")
        print(f"Средняя длина: {np.mean(history_results):.2f}")
        print(f"Длина гарантированного цикла: {cycle_length:.2f}")
    else:
        print("Не удалось найти допустимых решений")