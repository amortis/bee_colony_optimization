import tsplib95
import numpy as np
import random

def load_tsplib_instance(filename):
    """Загружает задачу из TSPLIB формата"""
    problem = tsplib95.load(filename)
    optimal = problem.optimal_value if hasattr(problem, 'optimal_value') else None
    
    nodes = list(problem.get_nodes())
    dimension = len(nodes)
    distance_matrix = np.zeros((dimension, dimension))
    
    for i in range(dimension):
        for j in range(dimension):
            distance_matrix[i][j] = problem._wfunc(i+1, j+1)
    
    return distance_matrix, optimal

def create_sparse_matrix_with_hamiltonian(distance_matrix, keep_probability=0.3, seed=None):
    """
    Преобразует полную матрицу в неполную, гарантируя существование гамильтонова цикла.
    
    Параметры:
    - distance_matrix: исходная полная матрица расстояний
    - keep_probability: вероятность сохранения ребра (0.3 = остаётся 30% рёбер)
    - seed: фиксирует случайность для воспроизводимости
    
    Возвращает:
    - sparse_matrix: разреженная матрица (отсутствующие рёбра = np.inf)
    - hamiltonian_cycle: гарантированный гамильтонов цикл
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    n = len(distance_matrix)
    sparse_matrix = np.full((n, n), np.inf)
    
    # 1. Создаём гарантированный гамильтонов цикл
    vertices = list(range(n))
    random.shuffle(vertices)  # случайный порядок городов
    
    # Добавляем рёбра цикла в матрицу
    for i in range(n):
        city_from = vertices[i]
        city_to = vertices[(i + 1) % n]
        sparse_matrix[city_from][city_to] = distance_matrix[city_from][city_to]
        sparse_matrix[city_to][city_from] = distance_matrix[city_to][city_from]
    
    # 2. Сохраняем дополнительные рёбра с заданной вероятностью
    for i in range(n):
        for j in range(i + 1, n):  # избегаем дублирования (i,j) и (j,i)
            # Пропускаем рёбра, уже добавленные в цикл
            if sparse_matrix[i][j] != np.inf:
                continue
                
            # С вероятностью keep_probability сохраняем ребро
            if random.random() < keep_probability:
                sparse_matrix[i][j] = distance_matrix[i][j]
                sparse_matrix[j][i] = distance_matrix[j][i]
    
    # 3. Убеждаемся, что граф связный (каждая вершина имеет хотя бы 2 связи)
    for i in range(n):
        connections = np.sum(sparse_matrix[i] != np.inf)
        if connections < 2:
            # Находим ближайшего соседа и добавляем ребро
            distances = distance_matrix[i].copy()
            distances[i] = np.inf  # исключаем диагональ
            
            # Ищем ближайшего, кого ещё нет в sparse_matrix
            for neighbor in np.argsort(distances):
                if sparse_matrix[i][neighbor] == np.inf:
                    sparse_matrix[i][neighbor] = distance_matrix[i][neighbor]
                    sparse_matrix[neighbor][i] = distance_matrix[neighbor][i]
                    break
    
    return sparse_matrix, vertices

def verify_hamiltonian(matrix, cycle):
    """Проверяет, является ли цикл допустимым гамильтоновым циклом"""
    total_distance = 0
    n = len(cycle)
    
    for i in range(n):
        city_from = cycle[i]
        city_to = cycle[(i + 1) % n]
        
        if matrix[city_from][city_to] == np.inf:
            print(f"⚠️ Нет ребра между {city_from} и {city_to}")
            return False, np.inf
        
        total_distance += matrix[city_from][city_to]
    
    return True, total_distance

def print_matrix_stats(matrix, original_matrix):
    """Выводит статистику по матрице"""
    n = len(matrix)
    total_possible = n * (n - 1)  # без диагонали
    existing_edges = np.sum(matrix != np.inf) // 2  # делим на 2 для неориентированного
    
    print(f"Городов: {n}")
    print(f"Рёбер в полном графе: {total_possible}")
    print(f"Рёбер в неполном графе: {existing_edges}")
    print(f"Плотность графа: {existing_edges/total_possible:.1%}")
    print(f"Средняя степень вершины: {2*existing_edges/n:.1f}")
    
    # Проверяем связность
    for i in range(n):
        connections = np.sum(matrix[i] != np.inf)
        if connections == 0:
            print(f"⚠️ Город {i} изолирован!")

# Пример использования
if __name__ == "__main__":
    # Загружаем вашу задачу
    matrix, optimal = load_tsplib_instance("matrix_task/st70.tsp")
    optimal = 675  # как у вас в коде
    
    print(f"Исходная задача: st70.tsp")
    print(f"Известный оптимум полного графа: {optimal}")
    print(f"Размер матрицы: {matrix.shape}\n")
    
    # Создаём неполную матрицу с гарантированным циклом
    sparse_matrix, hamiltonian_cycle = create_sparse_matrix_with_hamiltonian(
        matrix, 
        keep_probability=0.25,  # сохраняем 25% рёбер (можно менять)
        seed=42  # для воспроизводимости
    )
    
    print("=" * 50)
    print("НЕПОЛНАЯ МАТРИЦА СОЗДАНА")
    print("=" * 50)
    
    # Выводим статистику
    print_matrix_stats(sparse_matrix, matrix)
    
    # Проверяем гарантированный цикл
    is_valid, cycle_length = verify_hamiltonian(sparse_matrix, hamiltonian_cycle)
    
    print(f"\nГарантированный гамильтонов цикл (первые 10 городов): {hamiltonian_cycle[:10]}...")
    
    if is_valid:
        print(f"✅ Цикл корректен, длина: {cycle_length:.2f}")
        print(f"   Отклонение от оптимума полного графа: +{cycle_length-optimal:.2f} ({((cycle_length/optimal)-1):.1%})")
    else:
        print(f"❌ Цикл содержит отсутствующие рёбра")
    
    # Пример как использовать в алгоритме
    print(f"\nДля использования в алгоритме:")
    print(f"1. Матрица доступна как sparse_matrix")
    print(f"2. Отсутствующие рёбра обозначены np.inf")
    print(f"3. Проверка в алгоритме: if sparse_matrix[i][j] != np.inf: ...")