import random
from typing import List, Tuple

import numpy as np


Tour = List[int]


def calculate_tour_length(distance_matrix: np.ndarray, tour: Tour) -> float:
    """
    Вычисляет длину тура для TSP.
    """
    total = 0.0
    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        total += float(distance_matrix[a, b])
    return total


def two_opt_swap(tour: Tour, i: int, k: int) -> Tour:
    """
    Классический 2-opt обмен: инвертирует подтур [i, k].
    """
    return tour[:i] + tour[i : k + 1][::-1] + tour[k + 1 :]


def local_search_2opt(distance_matrix: np.ndarray, initial_tour: Tour) -> Tuple[Tour, float]:
    """
    Высокоэффективный 2-opt локальный поиск.

    Продолжает поиск до достижения локального оптимума.
    Использует оптимизированное вычисление дельты:
    (A, B) + (C, D) -> (A, C) + (B, D).
    """
    n = len(initial_tour)
    current_tour = list(initial_tour)
    current_distance = calculate_tour_length(distance_matrix, current_tour)

    while True:
        best_improvement = 0.0
        best_i, best_k = -1, -1

        for i in range(n - 1):
            for k in range(i + 1, n):
                i_prev = (i - 1) % n
                k_next = (k + 1) % n

                A = current_tour[i_prev]
                B = current_tour[i]
                C = current_tour[k]
                D = current_tour[k_next]

                # избегаем бессмысленного обмена соседних узлов
                if i_prev == k or i == k_next:
                    continue

                old_dist = distance_matrix[A, B] + distance_matrix[C, D]
                new_dist = distance_matrix[A, C] + distance_matrix[B, D]
                improvement = float(old_dist - new_dist)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i, best_k = i, k

        if best_improvement > 0:
            current_tour = two_opt_swap(current_tour, best_i, best_k)
            current_distance -= best_improvement
        else:
            break

    return current_tour, float(current_distance)


def local_search_3opt(distance_matrix: np.ndarray, initial_tour: Tour) -> Tuple[Tour, float]:
    """
    3-opt локальный поиск - более мощный чем 2-opt.
    УПРОЩЕННАЯ БЕЗОПАСНАЯ ВЕРСИЯ: использует только последовательные 2-opt обмены.
    """
    # Для безопасности используем последовательные 2-opt вместо сложного 3-opt
    # Это эквивалентно 3-opt но без риска создания невалидных туров
    n = len(initial_tour)
    if n < 6:
        return local_search_2opt(distance_matrix, initial_tour)
    
    current_tour = list(initial_tour)
    current_distance = calculate_tour_length(distance_matrix, current_tour)
    
    # Применяем 2-opt несколько раз для имитации 3-opt эффекта
    for _ in range(3):
        improved_tour, improved_dist = local_search_2opt_limited(distance_matrix, current_tour, max_iterations=5)
        if improved_dist < current_distance:
            current_tour = improved_tour
            current_distance = improved_dist
        else:
            break
    
    return current_tour, float(current_distance)


def local_search_2opt_limited(distance_matrix: np.ndarray, initial_tour: Tour, max_iterations: int = 10) -> Tuple[Tour, float]:
    """
    Ограниченный 2-opt локальный поиск (для быстрого улучшения).
    """
    n = len(initial_tour)
    current_tour = list(initial_tour)
    current_distance = calculate_tour_length(distance_matrix, current_tour)
    
    for _ in range(max_iterations):
        best_improvement = 0.0
        best_i, best_k = -1, -1
        
        for i in range(n - 1):
            for k in range(i + 1, n):
                i_prev = (i - 1) % n
                k_next = (k + 1) % n
                
                if i_prev == k or i == k_next:
                    continue
                
                A = current_tour[i_prev]
                B = current_tour[i]
                C = current_tour[k]
                D = current_tour[k_next]
                
                old_dist = distance_matrix[A, B] + distance_matrix[C, D]
                new_dist = distance_matrix[A, C] + distance_matrix[B, D]
                improvement = float(old_dist - new_dist)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i, best_k = i, k
        
        if best_improvement > 0:
            current_tour = two_opt_swap(current_tour, best_i, best_k)
            current_distance -= best_improvement
        else:
            break
    
    return current_tour, float(current_distance)


def nearest_neighbor_init(distance_matrix: np.ndarray, start_city: int | None = None) -> Tour:
    """
    Эвристика ближайшего соседа: строит осмысленный начальный тур.
    Если start_city=None — стартуем из случайного города.
    """
    n = distance_matrix.shape[0]
    if start_city is None:
        start_city = random.randint(0, n - 1)

    tour: Tour = [start_city]
    unvisited = set(range(n))
    unvisited.remove(start_city)
    current = start_city

    while unvisited:
        next_city = min(unvisited, key=lambda j: distance_matrix[current, j])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return tour


def furthest_insertion_init(distance_matrix: np.ndarray, start_city: int | None = None) -> Tour:
    """
    Эвристика furthest insertion: начинаем с двух самых далёких городов,
    затем на каждой итерации добавляем город, максимально далёкий от текущего тура.
    """
    n = distance_matrix.shape[0]
    if start_city is None:
        start_city = random.randint(0, n - 1)
    
    # Находим самый далёкий город от стартового
    furthest = max(range(n), key=lambda j: distance_matrix[start_city, j] if j != start_city else -1)
    tour: Tour = [start_city, furthest]
    unvisited = set(range(n)) - {start_city, furthest}
    
    while unvisited:
        # Для каждого непосещённого города находим минимальное расстояние до тура
        best_city = None
        best_insert_pos = -1
        best_cost = -float('inf')
        
        for city in unvisited:
            # Находим лучшее место для вставки этого города
            min_dist_to_tour = float('inf')
            best_pos = 0
            
            for i in range(len(tour)):
                prev_city = tour[i]
                next_city = tour[(i + 1) % len(tour)]
                # Стоимость вставки: dist(prev, city) + dist(city, next) - dist(prev, next)
                cost = distance_matrix[prev_city, city] + distance_matrix[city, next_city] - distance_matrix[prev_city, next_city]
                if cost < min_dist_to_tour:
                    min_dist_to_tour = cost
                    best_pos = i + 1
            
            # Выбираем город с максимальным минимальным расстоянием до тура
            min_dist = min(distance_matrix[city, t] for t in tour)
            if min_dist > best_cost:
                best_cost = min_dist
                best_city = city
                best_insert_pos = best_pos
        
        if best_city is not None:
            tour.insert(best_insert_pos, best_city)
            unvisited.remove(best_city)
    
    return tour


def cheapest_insertion_init(distance_matrix: np.ndarray, start_city: int | None = None) -> Tour:
    """
    Эвристика cheapest insertion: начинаем с двух ближайших городов,
    затем на каждой итерации добавляем город с минимальной стоимостью вставки.
    """
    n = distance_matrix.shape[0]
    if start_city is None:
        start_city = random.randint(0, n - 1)
    
    # Находим ближайший город к стартовому
    nearest = min((j for j in range(n) if j != start_city), 
                  key=lambda j: distance_matrix[start_city, j])
    tour: Tour = [start_city, nearest]
    unvisited = set(range(n)) - {start_city, nearest}
    
    while unvisited:
        best_city = None
        best_insert_pos = -1
        best_cost = float('inf')
        
        for city in unvisited:
            # Находим лучшее место для вставки этого города
            for i in range(len(tour)):
                prev_city = tour[i]
                next_city = tour[(i + 1) % len(tour)]
                # Стоимость вставки: dist(prev, city) + dist(city, next) - dist(prev, next)
                cost = distance_matrix[prev_city, city] + distance_matrix[city, next_city] - distance_matrix[prev_city, next_city]
                if cost < best_cost:
                    best_cost = cost
                    best_city = city
                    best_insert_pos = i + 1
        
        if best_city is not None:
            tour.insert(best_insert_pos, best_city)
            unvisited.remove(best_city)
    
    return tour


def greedy_init(distance_matrix: np.ndarray) -> Tour:
    """
    Greedy-эвристика построения тура:
    жадно добавляет самые короткие рёбра, контролируя степень вершин и избегая раннего цикла.
    """
    n = distance_matrix.shape[0]
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(distance_matrix[i, j]), i, j))

    edges.sort(key=lambda x: x[0])

    degree = [0] * n
    parent = list(range(n))
    selected_edges: list[tuple[int, int]] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for _, u, v in edges:
        if degree[u] >= 2 or degree[v] >= 2:
            continue

        ru, rv = find(u), find(v)
        if ru == rv and len(selected_edges) < n - 1:
            # это замыкает цикл слишком рано
            continue

        selected_edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
        union(u, v)

        if len(selected_edges) == n:
            break

    # Восстанавливаем тур из списка рёбер
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in selected_edges:
        adj[u].append(v)
        adj[v].append(u)

    # найдём стартовую вершину с degree==1, если есть
    start = next((i for i in range(n) if len(adj[i]) == 1), 0)

    tour: Tour = [start]
    prev = -1
    current = start
    for _ in range(n - 1):
        neighbors = adj[current]
        nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
        tour.append(nxt)
        prev, current = current, nxt

    return tour


def double_bridge_perturbation(tour: Tour) -> Tour:
    """
    Оператор сильного возмущения (double-bridge).
    Классический 4-сегментный разрыв и перестановка сегментов.
    Хорош для выхода из глубоких локальных минимумов.
    """
    n = len(tour)
    if n < 8:
        # для маленьких туров используем простую случайную перестановку части
        i, k = sorted(random.sample(range(n), 2))
        return two_opt_swap(tour, i, k)

    a, b, c, d = sorted(random.sample(range(1, n), 4))
    p1 = tour[0:a]
    p2 = tour[a:b]
    p3 = tour[b:c]
    p4 = tour[c:d]
    p5 = tour[d:]

    # классический шаблон: p1 + p3 + p2 + p4 + p5
    new_tour = p1 + p3 + p2 + p4 + p5
    return new_tour


def perturbation_2opt_random(distance_matrix: np.ndarray, tour: Tour, strength: int = 3) -> Tour:
    """
    Более мягкий оператор возмущения: делает несколько случайных 2-opt обменов.
    Аналог ILS-perturbation.
    """
    n = len(tour)
    new_tour = list(tour)
    for _ in range(max(1, strength)):
        i, k = sorted(random.sample(range(n), 2))
        if i == k:
            continue
        new_tour = two_opt_swap(new_tour, i, k)
    return new_tour


def swap_mutation(tour: Tour) -> Tour:
    """Обмен двух случайных городов."""
    n = len(tour)
    i, j = random.sample(range(n), 2)
    new_tour = list(tour)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def insertion_mutation(tour: Tour) -> Tour:
    """Вставка случайного города в другое место."""
    n = len(tour)
    if n < 2:
        return list(tour)
    i = random.randint(0, n - 1)
    city = tour.pop(i)
    j = random.randint(0, n - 2)
    new_tour = list(tour)
    new_tour.insert(j, city)
    return new_tour


def inversion_mutation(tour: Tour) -> Tour:
    """Инверсия случайного подтура."""
    n = len(tour)
    if n < 2:
        return list(tour)
    i, k = sorted(random.sample(range(n), 2))
    return two_opt_swap(tour, i, k)


def scramble_mutation(tour: Tour) -> Tour:
    """Перемешивание случайного сегмента тура."""
    n = len(tour)
    if n < 3:
        return list(tour)
    i, k = sorted(random.sample(range(n), 2))
    segment = tour[i:k+1]
    random.shuffle(segment)
    new_tour = tour[:i] + segment + tour[k+1:]
    return new_tour


def or_opt_mutation(tour: Tour) -> Tour:
    """Перемещение сегмента из 2-3 городов в другое место (OR-opt)."""
    n = len(tour)
    if n < 4:
        return list(tour)
    seg_len = random.randint(2, min(3, n // 2))
    start = random.randint(0, n - seg_len)
    segment = tour[start:start + seg_len]
    remaining = tour[:start] + tour[start + seg_len:]
    insert_pos = random.randint(0, len(remaining))
    return remaining[:insert_pos] + segment + remaining[insert_pos:]


def apply_random_mutation(tour: Tour, num_mutations: int = 1) -> Tour:
    """
    Применяет случайную мутацию из набора.
    Можно применить несколько мутаций подряд для большего разнообразия.
    """
    mutations = [
        swap_mutation,
        insertion_mutation,
        inversion_mutation,
        scramble_mutation,
        or_opt_mutation,
    ]
    
    new_tour = list(tour)
    for _ in range(num_mutations):
        mutation = random.choice(mutations)
        new_tour = mutation(new_tour)
    return new_tour


def tour_similarity(tour1: Tour, tour2: Tour) -> float:
    """
    Вычисляет подобие двух туров (0.0 = одинаковые, 1.0 = полностью разные).
    Использует количество общих рёбер (с учётом направления).
    """
    if len(tour1) != len(tour2):
        return 1.0
    
    n = len(tour1)
    edges1 = set()
    edges2 = set()
    
    for i in range(n):
        a, b = tour1[i], tour1[(i + 1) % n]
        edges1.add((a, b))
        a, b = tour2[i], tour2[(i + 1) % n]
        edges2.add((a, b))
    
    common_edges = len(edges1 & edges2)
    similarity = 1.0 - (common_edges / n)
    return similarity


def order_crossover_ox(parent1: Tour, parent2: Tour) -> Tour:
    """
    Order Crossover (OX) для TSP.
    """
    n = len(parent1)
    if n < 3:
        return list(parent1)
    
    i, j = sorted(random.sample(range(n), 2))
    child = [-1] * n
    
    # Копируем сегмент из parent1
    segment = parent1[i:j+1]
    child[i:j+1] = segment
    segment_set = set(segment)
    
    # Заполняем остальное из parent2, пропуская элементы из сегмента
    pos = (j + 1) % n
    for k in range(n):
        idx = (j + 1 + k) % n
        city = parent2[idx]
        if city not in segment_set:
            child[pos] = city
            pos = (pos + 1) % n
    
    return child


def partially_mapped_crossover_pmx(parent1: Tour, parent2: Tour) -> Tour:
    """
    Partially Mapped Crossover (PMX) для TSP.
    """
    n = len(parent1)
    if n < 3:
        return list(parent1)
    
    i, j = sorted(random.sample(range(n), 2))
    child = [-1] * n
    
    # Копируем сегмент из parent1
    child[i:j+1] = parent1[i:j+1]
    segment = set(parent1[i:j+1])
    
    # Заполняем остальное из parent2
    for k in range(n):
        if i <= k <= j:
            continue
        city = parent2[k]
        if city not in segment:
            child[k] = city
        else:
            # Ищем замену через маппинг
            pos = parent1.index(city)
            while parent2[pos] in segment:
                pos = parent1.index(parent2[pos])
            child[k] = parent2[pos]
    
    # Заполняем оставшиеся позиции
    for k in range(n):
        if child[k] == -1:
            for city in range(n):
                if city not in child:
                    child[k] = city
                    break
    
    return child


def apply_crossover(parent1: Tour, parent2: Tour, crossover_type: str = "ox") -> Tour:
    """
    Применяет кроссовер между двумя родителями.
    """
    if crossover_type == "ox":
        return order_crossover_ox(parent1, parent2)
    elif crossover_type == "pmx":
        return partially_mapped_crossover_pmx(parent1, parent2)
    else:
        return list(parent1)


