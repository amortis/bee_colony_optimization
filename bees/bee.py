from abc import ABC, abstractmethod


class Bee(ABC):
    """
    Абстрактный класс, представляющий пчелу в алгоритме пчелиной колонии.
    Все типы пчел (рабочие, наблюдатели, разведчики) наследуются от этого класса.
    """

    def __init__(self, solution, fitness_function, initial_fitness=None):
        """
        Инициализация пчелы.

        :param solution: Текущее решение пчелы (например, маршрут в задаче коммивояжёра).
        :param fitness_function: Функция, которая оценивает качество решения (фитнес-функция).
        """
        self.solution = solution
        self.fitness_function = fitness_function
        if initial_fitness is not None:
            self.fitness = initial_fitness
        else:
            self.fitness = self.calculate_fitness()
        # Счетчик неудачных исследований
        self.trial = 0

    def calculate_fitness(self):
        """
        Вычисляет фитнес текущего решения.

        :return: Значение фитнес-функции для текущего решения.
        """
        return self.fitness_function(self.solution)

    @abstractmethod
    def explore(self, other_solutions):
        """
        Абстрактный метод, который должен быть реализован в подклассах.
        Пчела исследует новые решения на основе текущего и других решений.

        :param other_solutions: Список других решений, которые могут быть использованы для исследования.
        """
        pass

    def update_solution(self, new_solution):
        """
        Обновляет текущее решение пчелы, если новое решение лучше.

        :param new_solution: Новое решение, которое предлагается для замены текущего.
        """
        new_fitness = self.fitness_function(new_solution)
        if new_fitness > self.fitness:
            self.solution = new_solution
            self.fitness = new_fitness
            return True
        return False

    # --- НОВЫЙ АБСТРАКТНЫЙ МЕТОД ДЛЯ АСИНХРОННОГО ВАРИАНТА ---
    @abstractmethod
    def explore_async(self, other_solutions, fitness_function):
        """
        Асинхронная версия explore, возвращает (is_improved, new_solution, new_fitness).
        """
        pass