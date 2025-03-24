from bees.bee import Bee
import random


class OnlookerBee(Bee):
    """
    Класс, представляющий пчелу-наблюдателя. Пчелы-наблюдатели выбирают решения на основе их качества (фитнеса).
    """

    def explore(self, other_solutions):
        """
        Пчела-наблюдатель выбирает решение на основе вероятности и пытается его улучшить.

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

        # Обновляем текущее решение, если новое лучше
        return self.update_solution(new_solution)

    def generate_new_solution(self, selected_solution):
        """
        Генерирует новое решение на основе выбранного решения.
        В задаче коммивояжёра это может быть перестановка двух городов в маршруте.

        :param selected_solution: Решение, выбранное на основе вероятности.
        :return: Новое решение.
        """
        # Копируем текущее решение, чтобы не изменять его напрямую
        new_solution = self.solution.copy()

        # Выбираем два случайных индекса для перестановки
        i, j = random.sample(range(len(new_solution)), 2)

        # Меняем местами два города в маршруте
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        return new_solution
