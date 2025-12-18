import random
import time
from datetime import timedelta
import numpy as np


class TSPSolver:
    """
    Оптимизированный решатель задачи коммивояжера (TSP)
    с использованием алгоритма Iterated Local Search (ILS)
    и высокоэффективного 2-opt локального поиска.
    """

    def __init__(self, distance_matrix, num_cities, max_time_seconds=60, seed=42):
        """
        Инициализация решателя.

        :param distance_matrix: Матрица расстояний между городами (numpy array или список списков).
        :param num_cities: Количество городов.
        :param max_time_seconds: Максимальное время работы алгоритма в секундах.
        :param seed: Начальное значение для генератора случайных чисел.
        """
        # Преобразуем в numpy array для эффективности
        self.distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        self.num_cities = num_cities
        self.max_time_seconds = max_time_seconds
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.best_tour = None
        self.best_distance = float('inf')
        self.history = []
        self.start_time = None
        self.end_time = None

    def _calculate_distance(self, tour):
        """ Вычисляет общую длину маршрута. """
        distance = 0
        for i in range(self.num_cities):
            city1 = tour[i]
            city2 = tour[(i + 1) % self.num_cities]
            distance += self.distance_matrix[city1, city2]
        return distance

    def _initial_tour(self):
        """ Создает начальный маршрут с использованием эвристики ближайшего соседа. """
        start_node = random.randint(0, self.num_cities - 1)
        tour = [start_node]
        unvisited = set(range(self.num_cities))
        unvisited.remove(start_node)

        current_node = start_node
        while unvisited:
            min_dist = float('inf')
            next_node = -1
            for neighbor in unvisited:
                dist = self.distance_matrix[current_node, neighbor]
                if dist < min_dist:
                    min_dist = dist
                    next_node = neighbor
            tour.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
        return tour

    def _two_opt_swap(self, tour, i, k):
        """ Выполняет 2-opt обмен (инверсию подмаршрута). """
        new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
        return new_tour

    def _local_search_2opt(self, initial_tour):
        """
        Высокоэффективный локальный поиск 2-opt.
        Продолжает поиск до тех пор, пока не будет найдено локально оптимальное решение.
        """
        current_tour = initial_tour
        current_distance = self._calculate_distance(current_tour)
        
        while True:
            best_improvement = 0
            best_i, best_k = -1, -1

            for i in range(self.num_cities - 1):
                for k in range(i + 1, self.num_cities):
                    # Оптимизированное вычисление изменения расстояния
                    # Узлы: ... -> A -> B -> ... -> C -> D -> ...
                    # Индексы: ... -> i-1 -> i -> ... -> k -> k+1 -> ...
                    
                    # Учитываем цикличность для i-1 и k+1
                    i_prev = (i - 1) % self.num_cities
                    k_next = (k + 1) % self.num_cities

                    # Города
                    A = current_tour[i_prev]
                    B = current_tour[i]
                    C = current_tour[k]
                    D = current_tour[k_next]

                    # Изменение расстояния: (новые ребра) - (старые ребра)
                    # Старые ребра: (A, B) + (C, D)
                    # Новые ребра: (A, C) + (B, D)
                    
                    # Избегаем обмена соседних узлов (i, k) = (i, i+1)
                    # В этом случае B и C - соседи, и 2-opt не имеет смысла
                    if i_prev == k or i == k_next:
                        continue

                    old_dist = self.distance_matrix[A, B] + self.distance_matrix[C, D]
                    new_dist = self.distance_matrix[A, C] + self.distance_matrix[B, D]
                    
                    improvement = old_dist - new_dist

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i, best_k = i, k

            if best_improvement > 0:
                # Выполняем лучший обмен
                current_tour = self._two_opt_swap(current_tour, best_i, best_k)
                current_distance -= best_improvement
            else:
                # Локальный оптимум достигнут
                break
        
        return current_tour, current_distance

    def _perturbation(self, tour, strength=3):
        """
        Оператор возмущения (перемешивания) для ILS.
        Выполняет 'strength' случайных 2-opt обменов.
        """
        new_tour = tour[:]
        for _ in range(strength):
            # Выбираем два случайных индекса
            i, k = sorted(random.sample(range(self.num_cities), 2))
            if i == k: 
                continue
            new_tour = self._two_opt_swap(new_tour, i, k)
        return new_tour

    def solve(self):
        """ Основной цикл Iterated Local Search. """
        self.start_time = time.time()
        
        # 1. Инициализация: Находим начальный локальный оптимум
        initial_tour = self._initial_tour()
        self.best_tour, self.best_distance = self._local_search_2opt(initial_tour)
        
        current_tour = self.best_tour
        current_distance = self.best_distance
        
        self.history.append(self.best_distance)
        
        print(f"Начальный локальный оптимум: {self.best_distance:.2f}")

        iteration = 0
        while time.time() - self.start_time < self.max_time_seconds:
            iteration += 1
            
            # 2. Возмущение (Perturbation)
            perturbation_strength = max(3, self.num_cities // 10)
            perturbed_tour = self._perturbation(current_tour, strength=perturbation_strength)
            
            # 3. Локальный поиск (Local Search)
            new_tour, new_distance = self._local_search_2opt(perturbed_tour)
            
            # 4. Критерий принятия (Acceptance Criterion)
            if new_distance < self.best_distance:
                # Найдено новое глобально лучшее решение
                self.best_distance = new_distance
                self.best_tour = new_tour
                current_tour = new_tour
                current_distance = new_distance
                print(f"Итерация {iteration}: Новое лучшее расстояние: {self.best_distance:.2f} (Время: {self.get_formatted_time()})")
            elif new_distance < current_distance:
                # Найдено лучшее решение в текущей траектории, но не глобально лучшее
                current_tour = new_tour
                current_distance = new_distance
            
            self.history.append(self.best_distance)
            
            if iteration % 100 == 0:
                print(f"Итерация {iteration}: Текущее лучшее расстояние: {self.best_distance:.2f} (Время: {self.get_formatted_time()})")

        self.end_time = time.time()
        print(f"\nРешение завершено. Итоговое расстояние: {self.best_distance:.2f}")
        return self.best_tour, self.best_distance

    def get_formatted_time(self, seconds=None):
        """ Форматирует время в читаемый вид (HH:MM:SS) """
        if seconds is None:
            seconds = time.time() - self.start_time if self.start_time else 0
        return str(timedelta(seconds=int(seconds))).split(".")[0]

    def get_elapsed_time(self):
        """ Возвращает прошедшее время в секундах """
        return time.time() - self.start_time if self.start_time else 0
