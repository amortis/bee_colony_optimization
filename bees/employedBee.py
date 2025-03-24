from bees.bee import Bee
import random


class EmployedBee(Bee):
    """
        Класс, представляющий рабочую пчелу. Рабочие пчелы отвечают за улучшение текущих решений.
    """
    def explore(self, other_solutions):
        """
        Рабочая пчела исследует окрестность текущего решения, пытаясь найти лучшее.

        :param other_solutions: Список других решений, которые могут быть использованы для исследования.
        :return: Возвращает True, если найдено лучшее решение, иначе False.
        """
        # Выбираем случайное решение из списка других решений
        partner_solution = random.choice(other_solutions)

        # Генерируем новое решение на основе текущего и партнерского
        new_solution = self.generate_new_solution(partner_solution)

        # Обновляем текущее решение, если новое лучше
        return self.update_solution(new_solution)

    def generate_new_solution(self, partner_solution):
        """
        Генерирует новое решение на основе текущего и партнерского решения.
        Перестановка двух городов в маршруте.

        :param partner_solution: Решение другой пчелы, используемое для генерации нового решения.
        :return: Новое решение.
        """
        # Меняем местами два случайных города в маршруте
        new_solution = self.solution.copy()
        i, j = random.sample(range(len(new_solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution
