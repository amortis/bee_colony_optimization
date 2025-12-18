"""
Оптимизированные функции для TSP с GPU и Numba ускорением.
"""
import random
from typing import List, Tuple
import numpy as np
from numba import jit, prange
import warnings

# Попытка импорта CuPy для GPU
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    try:
        import cupy as cp
        GPU_AVAILABLE = True
    except ImportError:
        GPU_AVAILABLE = False
        cp = None

Tour = List[int]


@jit(nopython=True, cache=True)
def calculate_tour_length_numba(distance_matrix: np.ndarray, tour: np.ndarray) -> float:
    """Numba-ускоренное вычисление длины тура."""
    total = 0.0
    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        total += distance_matrix[a, b]
    return total


@jit(nopython=True, cache=True, parallel=True)
def calculate_tour_lengths_batch_numba(distance_matrix: np.ndarray, tours: np.ndarray) -> np.ndarray:
    """Пакетное вычисление длин туров с Numba параллелизмом."""
    n_tours = tours.shape[0]
    n_cities = tours.shape[1]
    distances = np.zeros(n_tours, dtype=np.float64)
    
    for i in prange(n_tours):
        total = 0.0
        for j in range(n_cities):
            a = tours[i, j]
            b = tours[i, (j + 1) % n_cities]
            total += distance_matrix[a, b]
        distances[i] = total
    
    return distances


@jit(nopython=True, cache=True)
def two_opt_swap_numba(tour: np.ndarray, i: int, k: int) -> np.ndarray:
    """Numba-ускоренный 2-opt обмен."""
    n = len(tour)
    new_tour = np.zeros(n, dtype=np.int32)
    # Копируем до i
    for j in range(i):
        new_tour[j] = tour[j]
    # Инвертируем сегмент [i, k]
    idx = 0
    for j in range(i, k + 1):
        new_tour[j] = tour[k - idx]
        idx += 1
    # Копируем после k
    for j in range(k + 1, n):
        new_tour[j] = tour[j]
    return new_tour


@jit(nopython=True, cache=True)
def local_search_2opt_numba(distance_matrix: np.ndarray, initial_tour: np.ndarray, max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
    """Высокоэффективный 2-opt локальный поиск с Numba."""
    n = len(initial_tour)
    current_tour = initial_tour.copy()
    current_distance = calculate_tour_length_numba(distance_matrix, current_tour)
    
    iteration = 0
    while iteration < max_iterations:
        best_improvement = 0.0
        best_i, best_k = -1, -1
        
        for i in range(n - 1):
            for k in range(i + 1, n):
                i_prev = (i - 1 + n) % n
                k_next = (k + 1) % n
                
                if i_prev == k or i == k_next:
                    continue
                
                A = current_tour[i_prev]
                B = current_tour[i]
                C = current_tour[k]
                D = current_tour[k_next]
                
                old_dist = distance_matrix[A, B] + distance_matrix[C, D]
                new_dist = distance_matrix[A, C] + distance_matrix[B, D]
                improvement = old_dist - new_dist
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i, best_k = i, k
        
        if best_improvement > 1e-9:
            current_tour = two_opt_swap_numba(current_tour, best_i, best_k)
            current_distance -= best_improvement
            iteration += 1
        else:
            break
    
    return current_tour, current_distance


@jit(nopython=True, cache=True)
def local_search_2opt_limited_numba(distance_matrix: np.ndarray, initial_tour: np.ndarray, max_iterations: int = 10) -> Tuple[np.ndarray, float]:
    """Ограниченный 2-opt локальный поиск с Numba."""
    n = len(initial_tour)
    current_tour = initial_tour.copy()
    current_distance = calculate_tour_length_numba(distance_matrix, current_tour)
    
    for iteration in range(max_iterations):
        best_improvement = 0.0
        best_i, best_k = -1, -1
        
        for i in range(n - 1):
            for k in range(i + 1, n):
                i_prev = (i - 1 + n) % n
                k_next = (k + 1) % n
                
                if i_prev == k or i == k_next:
                    continue
                
                A = current_tour[i_prev]
                B = current_tour[i]
                C = current_tour[k]
                D = current_tour[k_next]
                
                old_dist = distance_matrix[A, B] + distance_matrix[C, D]
                new_dist = distance_matrix[A, C] + distance_matrix[B, D]
                improvement = old_dist - new_dist
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i, best_k = i, k
        
        if best_improvement > 1e-9:
            current_tour = two_opt_swap_numba(current_tour, best_i, best_k)
            current_distance -= best_improvement
        else:
            break
    
    return current_tour, current_distance


def calculate_tour_lengths_batch_gpu(distance_matrix_gpu: cp.ndarray, tours: List[Tour]) -> List[float]:
    """Пакетное вычисление длин туров на GPU (полностью векторизовано)."""
    if not GPU_AVAILABLE or distance_matrix_gpu is None or len(tours) < 1:
        return []
    
    try:
        n = len(tours)
        num_cities = len(tours[0])
        
        # Преобразуем туры в GPU массив
        tours_gpu = cp.asarray(tours, dtype=cp.int32)  # shape: (n, num_cities)
        
        # Векторизованное вычисление: для каждого тура суммируем расстояния
        # tours_gpu[:, i] - текущий город для всех туров
        # tours_gpu[:, (i+1) % num_cities] - следующий город для всех туров
        distances = cp.zeros(n, dtype=cp.float64)
        
        for i in range(num_cities):
            current_cities = tours_gpu[:, i]
            next_cities = tours_gpu[:, (i + 1) % num_cities]
            distances += distance_matrix_gpu[current_cities, next_cities]
        
        return cp.asnumpy(distances).tolist()
    except Exception as e:
        print(f"GPU batch calculation error: {e}")
        return []


def calculate_tour_length_gpu(distance_matrix_gpu: cp.ndarray, tour: Tour) -> float:
    """Вычисление длины тура на GPU."""
    if not GPU_AVAILABLE or distance_matrix_gpu is None:
        return 0.0
    
    try:
        tour_gpu = cp.asarray(tour, dtype=cp.int32)
        n = len(tour)
        total = cp.float64(0.0)
        
        for i in range(n):
            a = tour_gpu[i]
            b = tour_gpu[(i + 1) % n]
            total += distance_matrix_gpu[a, b]
        
        return float(total)
    except Exception:
        return 0.0


# Обёртки для совместимости с существующим кодом
def calculate_tour_length(distance_matrix: np.ndarray, tour: Tour) -> float:
    """Вычисляет длину тура (использует Numba если возможно)."""
    tour_np = np.asarray(tour, dtype=np.int32)
    return calculate_tour_length_numba(distance_matrix, tour_np)


def local_search_2opt(distance_matrix: np.ndarray, initial_tour: Tour, max_iterations: int = 1000) -> Tuple[Tour, float]:
    """2-opt локальный поиск (использует Numba)."""
    tour_np = np.asarray(initial_tour, dtype=np.int32)
    improved_tour, improved_dist = local_search_2opt_numba(distance_matrix, tour_np, max_iterations)
    return improved_tour.tolist(), float(improved_dist)


def local_search_2opt_limited(distance_matrix: np.ndarray, initial_tour: Tour, max_iterations: int = 10) -> Tuple[Tour, float]:
    """Ограниченный 2-opt локальный поиск (использует Numba)."""
    tour_np = np.asarray(initial_tour, dtype=np.int32)
    improved_tour, improved_dist = local_search_2opt_limited_numba(distance_matrix, tour_np, max_iterations)
    return improved_tour.tolist(), float(improved_dist)

