import random
import numpy as np

# --- Вспомогательные функции для TSP ---

def calculate_distance(tour, distance_matrix):
    """ Вычисляет общую длину маршрута. """
    num_cities = len(tour)
    distance = 0
    for i in range(num_cities):
        city1 = tour[i]
        city2 = tour[(i + 1) % num_cities]
        distance += distance_matrix[city1, city2]
    return distance

def two_opt_swap(tour, i, k):
    """ Выполняет 2-opt обмен (инверсию подмаршрута). """
    new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    return new_tour

def local_search_2opt(initial_tour, distance_matrix):
    """
    Высокоэффективный локальный поиск 2-opt.
    Продолжает поиск до тех пор, пока не будет найдено локально оптимальное решение.
    """
    num_cities = len(initial_tour)
    current_tour = initial_tour
    current_distance = calculate_distance(current_tour, distance_matrix)
    
    while True:
        best_improvement = 0
        best_i, best_k = -1, -1

        for i in range(num_cities - 1):
            for k in range(i + 1, num_cities):
                # Оптимизированное вычисление изменения расстояния
                i_prev = (i - 1) % num_cities
                k_next = (k + 1) % num_cities

                A = current_tour[i_prev]
                B = current_tour[i]
                C = current_tour[k]
                D = current_tour[k_next]

                # Избегаем обмена соседних узлов
                if i_prev == k or i == k_next:
                    continue

                old_dist = distance_matrix[A, B] + distance_matrix[C, D]
                new_dist = distance_matrix[A, C] + distance_matrix[B, D]
                
                improvement = old_dist - new_dist

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i, best_k = i, k

        if best_improvement > 0:
            # Выполняем лучший обмен
            current_tour = two_opt_swap(current_tour, best_i, best_k)
            current_distance -= best_improvement
        else:
            # Локальный оптимум достигнут
            break
    
    return current_tour, current_distance

# --- Эвристики для инициализации ---

def nearest_neighbor_init(distance_matrix, random_start=False):
    """ Создает маршрут с использованием эвристики ближайшего соседа. """
    num_cities = distance_matrix.shape[0]
    if random_start:
        start_node = random.randint(0, num_cities - 1)
    else:
        # Начинаем с города 0 для детерминированности, если не указан случайный старт
        start_node = 0 
        
    tour = [start_node]
    unvisited = set(range(num_cities))
    unvisited.remove(start_node)

    current_node = start_node
    while unvisited:
        min_dist = float('inf')
        next_node = -1
        for neighbor in unvisited:
            dist = distance_matrix[current_node, neighbor]
            if dist < min_dist:
                min_dist = dist
                next_node = neighbor
        tour.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node
    return tour

def greedy_init(distance_matrix):
    """ Создает маршрут с использованием жадной эвристики (самое короткое ребро). """
    num_cities = distance_matrix.shape[0]
    
    # Создаем список всех ребер (расстояние, город1, город2)
    edges = []
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            edges.append((distance_matrix[i, j], i, j))
            
    # Сортируем ребра по расстоянию
    edges.sort()
    
    # Инициализируем структуру для отслеживания степени узлов и циклов
    parent = list(range(num_cities))
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    # Строим маршрут
    tour_edges = []
    degree = [0] * num_cities
    
    for dist, u, v in edges:
        # Проверяем, не превысит ли степень узла 2
        if degree[u] < 2 and degree[v] < 2:
            # Проверяем, не создаст ли это цикл, если количество ребер меньше N
            if find(u) != find(v) or len(tour_edges) == num_cities - 1:
                union(u, v)
                tour_edges.append((u, v))
                degree[u] += 1
                degree[v] += 1
                
                if len(tour_edges) == num_cities:
                    break
    
    # Преобразуем список ребер в маршрут (последовательность городов)
    adj = {i: [] for i in range(num_cities)}
    for u, v in tour_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Находим начальный узел (любой)
    start_node = tour_edges[0][0]
    
    # Проходим по маршруту
    tour = []
    current = start_node
    visited = {start_node}
    
    for _ in range(num_cities):
        tour.append(current)
        
        found_next = False
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                current = neighbor
                found_next = True
                break
        
        if not found_next and len(tour) < num_cities:
            # Находим последний узел, который не был посещен
            for i in range(num_cities):
                if i not in visited:
                    current = i
                    tour.append(current)
                    break
            break # Должен быть цикл, но для безопасности
            
    return tour
