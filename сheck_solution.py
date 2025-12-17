import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matrix_task.sparred_tsp_task import load_tsplib_instance, create_sparse_matrix_with_hamiltonian

def analyze_solution(matrix, solution):
    """
    Анализирует решение на правильность
    """
    n = len(matrix)
    print("\n=== Анализ решения ===")
    
    # 1. Проверка, что все города присутствуют
    cities_in_solution = set(solution)
    print(f"Городов в решении: {len(cities_in_solution)}")
    print(f"Всего должно быть городов: {n}")
    
    if len(cities_in_solution) != n:
        print("❌ ОШИБКА: Не все города посещены!")
        missing = set(range(n)) - cities_in_solution
        print(f"Пропущенные города: {sorted(missing)}")
        return False
    else:
        print("✅ Все города посещены")
    
    # 2. Проверка дубликатов
    if len(solution) != len(cities_in_solution):
        print("❌ ОШИБКА: Есть повторяющиеся города!")
        return False
    else:
        print("✅ Нет повторяющихся городов")
    
    # 3. Проверка существования всех рёбер
    total_distance = 0
    valid = True
    
    for i in range(n):
        city_from = solution[i]
        city_to = solution[(i + 1) % n]
        
        if np.isinf(matrix[city_from][city_to]):
            print(f"❌ ОШИБКА: Отсутствует ребро между {city_from} и {city_to}")
            valid = False
        else:
            total_distance += matrix[city_from][city_to]
    
    if valid:
        print(f"✅ Все рёбра существуют")
        print(f"✅ Общая длина маршрута: {total_distance:.2f}")
        return True, total_distance
    else:
        print("❌ Решение содержит отсутствующие рёбра")
        return False, None

# Загружаем TSP задачу
print("Загрузка TSP задачи...")
full_matrix, optimal = load_tsplib_instance("matrix_task/st70.tsp")

# Создаем неполный граф с гарантированным циклом
print("Создание неполного графа...")
DISTANCE_MATRIX, hamiltonian_cycle = create_sparse_matrix_with_hamiltonian(
    full_matrix, 
    keep_probability=0.3,  # Оставляем 30% рёбер
    seed=42
)



solution = [15, 64, 50, 55, 49, 66, 63, 61, 10, 20, 60, 38, 24, 39, 8, 5, 14, 23, 65, 4, 51, 9, 52, 46, 36, 57, 47, 32, 11, 33, 44, 45, 53, 16, 59, 42, 43, 40, 3, 6, 41, 7, 67, 26, 29, 13, 31, 17, 1, 2, 27, 48, 54, 19, 25, 18, 56, 34, 30, 68, 69, 12, 58, 62, 21, 37, 28, 35, 0, 22]


print(analyze_solution(DISTANCE_MATRIX, solution))