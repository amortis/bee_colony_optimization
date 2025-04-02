from bees.bee import Bee
import random


class EmployedBee(Bee):
    """
        Класс, представляющий рабочую пчелу. Рабочие пчелы отвечают за улучшение текущих решений.
    """
    def explore(self, other_solutions):
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
                while partner_solution[ptr] in new_solution:
                    ptr += 1
                new_solution[i] = partner_solution[ptr]

        return new_solution
