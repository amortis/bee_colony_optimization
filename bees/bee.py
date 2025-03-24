from abc import ABC, abstractmethod


class Bee(ABC):
    """
    Абстрактный класс, представляющий пчелу в алгоритме пчелиной колонии.
    Все типы пчел (рабочие, наблюдатели, разведчики) наследуются от этого класса.
    """

    def __init__(self, solution, fitness_function):
        """
        Инициализация пчелы.

        :param solution: Текущее решение пчелы (например, маршрут в задаче коммивояжёра).
        :param fitness_function: Функция, которая оценивает качество решения (фитнес-функция).
        """
        self.solution = solution
        self.fitness_function = fitness_function
        self.fitness = self.calculate_fitness()

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
