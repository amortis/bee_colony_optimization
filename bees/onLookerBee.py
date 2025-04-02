from bees.bee import Bee
import random


class OnlookerBee(Bee):
    """
    Класс, представляющий пчелу-наблюдателя. Пчелы-наблюдатели выбирают решения на основе их качества (фитнеса).
    """

    def explore(self, other_solutions):
        """
        Пчела-наблюдатель выбирает решение на основе вероятности и пытается его улучшить.
        Исследует окрестность выбранного решения.

        :param other_solutions: Список других решений.
        :return: Возвращает True, если найдено лучшее решение, иначе False.
        """
        # Вычисляем фитнес для каждого решения
        fitness_values = [self.fitness_function(sol) for sol in other_solutions]

        # Вычисляем вероятности выбора каждого решения
        total_fitness = sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values] #Pi

        # Выбираем решение на основе вероятности
        selected_solution = random.choices(other_solutions, weights=probabilities, k=1)[0]

        # Генерируем новое решение на основе выбранного
        new_solution = self.generate_new_solution(selected_solution)

        # Жадный выбор с обновлением trial
        new_fitness = self.fitness_function(new_solution)
        if new_fitness > self.fitness:
            self.solution = new_solution
            self.fitness = new_fitness
            self.trial = 0  # Сброс счетчика при успехе
            return True
        else:
            self.trial += 1  # Увеличение счетчика при неудаче
            return False

    def generate_new_solution(self, selected_solution):
        """
        Генерирует новое решение на основе выбранного решения.
        Аналогично EmployedBee, но работает с выбранным (а не случайным) решением.

        :param selected_solution: Решение, выбранное на основе вероятности.
        :return: Новое решение.
        """
        new_solution = self.solution.copy()

        # Выбираем случайный сегмент из текущего решения
        start = random.randint(0, len(self.solution) - 2)
        end = random.randint(start + 1, len(self.solution))

        # Копируем сегмент
        new_solution[start:end] = self.solution[start:end]

        # Заполняем остальное из выбранного решения
        ptr = 0
        for i in range(len(new_solution)):
            if new_solution[i] == -1:
                while selected_solution[ptr] in new_solution:
                    ptr += 1
                new_solution[i] = selected_solution[ptr]

        return new_solution
