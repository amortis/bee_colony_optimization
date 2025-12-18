import random
from typing import List, Tuple, Optional

import numpy as np

# Попытка импортировать Numba, если доступна
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Заглушка для случая, когда Numba недоступна
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


Tour = List[int]


# Numba-версия calculate_tour_length (быстрая)
if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def _calculate_tour_length_numba(dist_matrix: np.ndarray, tour: np.ndarray) -> float:
        """Numba-компилированная версия вычисления длины тура."""
        length = 0.0
        n = len(tour)
        for i in range(n - 1):
            length += dist_matrix[tour[i], tour[i+1]]
        length += dist_matrix[tour[-1], tour[0]]
        return length


def calculate_tour_length(distance_matrix: np.ndarray, tour: Tour) -> float:
    """
    Вычисляет длину тура для TSP.
    Использует Numba-версию, если доступна, иначе Python-версию.
    """
    if NUMBA_AVAILABLE and isinstance(tour, np.ndarray):
        return _calculate_tour_length_numba(distance_matrix, tour)
    
    # Python-версия (fallback)
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


def delta_2opt_evaluation(distance_matrix: np.ndarray, tour: Tour, i: int, j: int) -> float:
    """
    Вычисляет изменение длины при 2-opt swap без полного пересчета.
    Возвращает разницу: new_distance - old_distance (отрицательное значение = улучшение).
    
    :param distance_matrix: Матрица расстояний
    :param tour: Текущий тур
    :param i: Индекс первого города (i < j)
    :param j: Индекс второго города
    :return: Изменение длины (отрицательное = улучшение)
    """
    n = len(tour)
    i_prev = (i - 1) % n
    i_next = (i + 1) % n
    j_prev = (j - 1) % n
    j_next = (j + 1) % n
    
    i_city = tour[i]
    j_city = tour[j]
    
    # Удаляемые ребра
    old_edges = (distance_matrix[tour[i_prev], i_city] + 
                 distance_matrix[i_city, tour[i_next]] +
                 distance_matrix[tour[j_prev], j_city] + 
                 distance_matrix[j_city, tour[j_next]])
    
    # Новые ребра после 2-opt swap
    new_edges = (distance_matrix[tour[i_prev], j_city] + 
                 distance_matrix[j_city, tour[i_next]] +
                 distance_matrix[tour[j_prev], i_city] + 
                 distance_matrix[i_city, tour[j_next]])
    
    return new_edges - old_edges


# Numba-версия fast_2opt (First Improvement стратегия)
if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def _fast_2opt_numba(dist_matrix: np.ndarray, tour: np.ndarray) -> np.ndarray:
        """
        Ускоренная версия 2-opt с использованием First Improvement (выход после первого улучшения).
        """
        n = len(tour)
        best_tour = tour.copy()
        improved = True
        
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue  # Не разрываем начало и конец
                    
                    # Оценка выигрыша (delta) без полного пересчета пути
                    # A-B ... C-D  ->  A-C ... B-D
                    a, b = best_tour[i], best_tour[i+1]
                    c, d = best_tour[j], best_tour[(j+1) % n]
                    
                    delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                    
                    if delta < -1e-9:
                        # Выполняем разворот сегмента
                        segment = best_tour[i+1:j+1]
                        best_tour[i+1:j+1] = segment[::-1]
                        improved = True
                        break  # First Improvement - выходим после первого улучшения
                if improved:
                    break
        
        return best_tour


def fast_2opt(distance_matrix: np.ndarray, tour) -> np.ndarray:
    """
    Быстрая версия 2-opt с First Improvement стратегией.
    Использует Numba, если доступна.
    
    :param distance_matrix: Матрица расстояний
    :param tour: Тур (может быть list или np.ndarray)
    :return: Оптимизированный тур как np.ndarray
    """
    tour_array = np.asarray(tour, dtype=np.int32)
    
    if NUMBA_AVAILABLE:
        return _fast_2opt_numba(distance_matrix, tour_array)
    else:
        # Python fallback
        return np.array(local_search_2opt(distance_matrix, list(tour_array))[0], dtype=np.int32)


def local_search_2opt(distance_matrix: np.ndarray, initial_tour: Tour) -> Tuple[Tour, float]:
    """
    Высокоэффективный 2-opt локальный поиск с delta-evaluation.
    Использует fast_2opt, если доступна Numba.

    Продолжает поиск до достижения локального оптимума.
    """
    # Пробуем использовать Numba-версию
    if NUMBA_AVAILABLE:
        tour_array = np.asarray(initial_tour, dtype=np.int32)
        optimized = _fast_2opt_numba(distance_matrix, tour_array)
        distance = calculate_tour_length(distance_matrix, optimized)
        return list(optimized), float(distance)
    
    # Python-версия (fallback)
    n = len(initial_tour)
    current_tour = list(initial_tour)
    current_distance = calculate_tour_length(distance_matrix, current_tour)

    while True:
        best_improvement = 0.0
        best_i, best_k = -1, -1

        for i in range(n - 1):
            for k in range(i + 1, n):
                # избегаем бессмысленного обмена соседних узлов
                if (i + 1) % n == k or (k + 1) % n == i:
                    continue
                
                # Используем delta-evaluation для быстрого вычисления изменения
                delta = delta_2opt_evaluation(distance_matrix, current_tour, i, k)
                improvement = -delta  # отрицательная дельта = улучшение

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


def local_search_3opt(distance_matrix: np.ndarray, initial_tour: Tour) -> Tuple[Tour, float]:
    """
    Локальный поиск 3-opt (более агрессивный, чем 2-opt).
    Используется когда 2-opt перестает находить улучшения.
    """
    n = len(initial_tour)
    current_tour = list(initial_tour)
    current_distance = calculate_tour_length(distance_matrix, current_tour)
    improved = True
    
    while improved:
        improved = False
        best_improvement = 0.0
        best_move = None
        
        # Перебираем все возможные 3-opt перестановки
        for i in range(n):
            for j in range(i + 2, n):
                for k in range(j + 2, n):
                    if k == n - 1 and i == 0:
                        continue  # Избегаем тривиальных случаев
                    
                    # 7 возможных способов пересоединения после удаления 3 рёбер
                    # Рассматриваем только несколько наиболее перспективных
                    moves = [
                        # Вариант 1: (i, i+1), (j, j+1), (k, k+1) -> (i, j+1), (k, i+1), (j, k+1)
                        (current_tour[i], current_tour[(i+1)%n], 
                         current_tour[j], current_tour[(j+1)%n],
                         current_tour[k], current_tour[(k+1)%n],
                         current_tour[i], current_tour[(j+1)%n],
                         current_tour[k], current_tour[(i+1)%n],
                         current_tour[j], current_tour[(k+1)%n]),
                    ]
                    
                    for move in moves:
                        old_dist = (distance_matrix[move[0], move[1]] + 
                                   distance_matrix[move[2], move[3]] + 
                                   distance_matrix[move[4], move[5]])
                        new_dist = (distance_matrix[move[6], move[7]] + 
                                   distance_matrix[move[8], move[9]] + 
                                   distance_matrix[move[10], move[11]])
                        improvement = old_dist - new_dist
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (i, j, k, move)
        
        if best_improvement > 0 and best_move is not None:
            # Применяем лучшее улучшение (упрощенная версия)
            # В полной реализации нужно перестроить тур
            # Здесь используем упрощенный подход: делаем 2-opt swap
            i, j, k, _ = best_move
            if j - i > 1:
                current_tour = two_opt_swap(current_tour, i, j)
            current_distance -= best_improvement
            improved = True
    
    return current_tour, float(current_distance)


def build_candidate_lists(distance_matrix: np.ndarray, num_neighbors: int = 20) -> np.ndarray:
    """
    Строит списки кандидатов (ближайших соседей) для каждого города.
    Это ускоряет 2-opt поиск, так как оптимальные рёбра обычно соединяют близкие города.
    
    :param distance_matrix: Матрица расстояний
    :param num_neighbors: Количество ближайших соседей для каждого города
    :return: Матрица размером (n_cities, num_neighbors) с индексами ближайших соседей
    """
    n = distance_matrix.shape[0]
    num_neighbors = min(num_neighbors, n - 1)
    candidate_lists = np.zeros((n, num_neighbors), dtype=np.int32)
    
    for i in range(n):
        # Получаем индексы всех городов, отсортированных по расстоянию
        distances = distance_matrix[i, :]
        # Исключаем сам город (расстояние = 0)
        sorted_indices = np.argsort(distances)[1:num_neighbors + 1]
        candidate_lists[i, :] = sorted_indices[:num_neighbors]
    
    return candidate_lists


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



