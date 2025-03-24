from bees.bee import Bee
import random


class ScoutBee(Bee):
    """
    Класс, представляющий пчелу-разведчика. Пчелы-разведчики ищут новые случайные решения.
    """

    def __init__(self, solution, fitness_function, lb, ub):
        """
        Инициализация пчелы-разведчика.

        :param solution: Текущее решение пчелы.
        :param fitness_function: Функция, которая оценивает качество решения.
        :param lb: Нижняя граница пространства решений.
        :param ub: Верхняя граница пространства решений.
        """
        super().__init__(solution, fitness_function)
        self.lb = lb
        self.ub = ub

    def explore(self, other_solutions):
        """
        Пчела-разведчик генерирует новое случайное решение в пределах заданных границ.

        :param other_solutions: Список других решений (не используется в ScoutBee).
        :return: Возвращает True, если найдено лучшее решение, иначе False.
        """
        # Генерируем новое случайное решение
        new_solution = self.generate_random_solution(self.lb, self.ub)

        # Обновляем текущее решение, если новое лучше
        return self.update_solution(new_solution)

    def generate_random_solution(self, lb, ub):
        """
        Генерирует случайное решение в пределах заданных границ.

        :param lb: Нижняя граница пространства решений.
        :param ub: Верхняя граница пространства решений.
        :return: Случайное решение.
        """
        # Случайный маршрут для задачи коммивояжёра
        return random.sample(range(lb, ub + 1), len(range(lb, ub + 1)))
