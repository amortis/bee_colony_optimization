from bees.bee import Bee
import random
from typing import Optional
import numpy as np


class EmployedBee(Bee):
    """
        Класс, представляющий рабочую пчелу. Рабочие пчелы отвечают за улучшение текущих решений.
    """
    
    def __init__(self, solution, fitness_function, initial_fitness=None, distance_matrix: Optional[np.ndarray] = None):
        """
        Инициализация рабочей пчелы с поддержкой кэширования длины тура.
        
        :param solution: Текущее решение пчелы
        :param fitness_function: Функция оценки качества решения
        :param initial_fitness: Начальное значение фитнеса (опционально)
        :param distance_matrix: Матрица расстояний для кэширования длины тура (опционально)
        """
        super().__init__(solution, fitness_function, initial_fitness)
        self.distance_matrix = distance_matrix
        self._cached_length: Optional[float] = None
    def explore(self, other_solutions): # type: ignore
        """
        Рабочая пчела исследует окрестность текущего решения, пытаясь найти лучшее.
        """
        # 1. Выбираем случайного партнера (исключая текущую пчелу)
        partner = random.choice([bee for bee in other_solutions if bee != self])

        # 2. Генерируем новое решение через комбинацию с партнером
        new_solution = self.generate_new_solution(partner.solution)

        # 3. Жадный выбор
        new_fitness = self.fitness_function(new_solution)
        if new_fitness > self.fitness:
            self.solution = new_solution
            self.fitness = new_fitness
            self.trial = 0
            self.invalidate_cache()  # Инвалидируем кэш при изменении решения
            return True
        else:
            self.trial += 1
            return False

    def generate_new_solution(self, partner_solution):
        """
        Генерирует новое решение на основе текущего и партнерского решения.
        1. Выбор случайного сегмента из текущего решения
        2. Заполнение остального из партнерского решения

        :param partner_solution: Решение другой пчелы, используемое для генерации нового решения.
        :return: Новое решение.
        """
        size = len(self.solution)
        new_solution = [-1] * size

        # Выбираем случайный отрезок (минимум 2 города)
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, min(start + size // 2, size))

        # Копируем сегмент из текущего решения
        new_solution[start:end] = self.solution[start:end]

        # Заполняем остальное из партнерского решения (порядок сохранен)
        ptr = 0
        for i in range(size):
            if new_solution[i] == -1:
                used = set(self.solution[start:end])  # Создаём множество для O(1) поиска
                while partner_solution[ptr] in new_solution:
                    ptr += 1
                new_solution[i] = partner_solution[ptr]

        return new_solution
    
    def get_tour_length(self, distance_matrix: np.ndarray) -> float:
        """
        Вычисляет длину тура с кэшированием.
        
        :param distance_matrix: Матрица расстояний
        :return: Длина тура
        """
        if self._cached_length is None or self.distance_matrix is not distance_matrix:
            from tsp_optimizations import calculate_tour_length
            self._cached_length = calculate_tour_length(distance_matrix, self.solution)
            self.distance_matrix = distance_matrix
        return self._cached_length
    
    def invalidate_cache(self):
        """Инвалидирует кэш длины тура (вызывать при изменении решения)"""
        self._cached_length = None
    
    def update_solution(self, new_solution):
        """
        Обновляет текущее решение пчелы, если новое решение лучше.
        Также инвалидирует кэш.
        """
        result = super().update_solution(new_solution)
        if result:
            self.invalidate_cache()
        return result

    # --- НОВАЯ АСИНХРОННАЯ ВЕРСИЯ ---
    def explore_async(self, other_solutions, fitness_function): # type: ignore
        """
        Асинхронная версия explore, возвращает (is_improved, new_solution, new_fitness).
        """
        partner = random.choice([bee for bee in other_solutions if bee != self])
        new_solution = self.generate_new_solution(partner.solution)
        new_fitness = fitness_function(new_solution) # Используем переданную функцию

        if new_fitness > self.fitness:
            return True, new_solution, new_fitness
        else:
            return False, self.solution, self.fitness # Возвращаем старые значения, если не улучшено
