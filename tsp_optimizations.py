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



