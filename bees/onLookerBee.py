import math

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
        if not other_solutions:
            return False

            # 1. Вычисляем вероятности выбора для каждого решения
        probabilities = self._calculate_selection_probabilities(other_solutions)

        # 2. Выбираем решение для исследования (на основе вероятностей)
        selected_solution = self._select_solution_based_on_probability(other_solutions, probabilities)

        # 3. Генерируем новое решение на основе выбранного
        new_solution = self._generate_new_solution(selected_solution)

        # 4. Проверяем и применяем улучшение
        return self._greedy_selection(new_solution)

    def _calculate_selection_probabilities(self, solutions):
        """Вычисляет вероятности выбора для каждого решения"""
        # Получаем значения фитнес-функции для всех решений
        fitness_values = [self.fitness_function(sol) for sol in solutions]

        # Преобразуем fitness в вероятности (используем softmax)
        max_fitness = max(fitness_values) if fitness_values else 1
        exp_values = [math.exp((f - max_fitness) * 10) for f in fitness_values]  # Масштабируем разницу
        total = sum(exp_values)

        return [exp / total for exp in exp_values] if total > 0 else [1 / len(solutions)] * len(solutions)

    def _select_solution_based_on_probability(self, solutions, probabilities):
        """Выбирает решение для исследования на основе вероятностей"""
        # Используем метод рулетки с 3 попытками для выбора лучшего
        candidates = random.choices(solutions, weights=probabilities, k=3)
        return max(candidates, key=lambda x: self.fitness_function(x))

    def _generate_new_solution(self, base_solution):
        """Генерирует модифицированное решение на основе базового"""
        new_solution = base_solution.copy()

        # Применяем комбинацию мутаций специально для TSP
        mutation_type = random.choice(["inversion", "swap", "shift"])

        if mutation_type == "inversion":
            # Инверсия случайного сегмента
            i, j = sorted(random.sample(range(len(new_solution)), 2))
            new_solution[i:j + 1] = reversed(new_solution[i:j + 1])
        elif mutation_type == "swap":
            # Обмен двух случайных городов
            i, j = random.sample(range(len(new_solution)), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        else:
            # Сдвиг случайного города
            city = new_solution.pop(random.randint(0, len(new_solution) - 1))
            new_solution.insert(random.randint(0, len(new_solution)), city)

        return new_solution

    def _greedy_selection(self, new_solution):
        """Применяет жадный выбор с обновлением состояния"""
        new_fitness = self.fitness_function(new_solution)

        if new_fitness > self.fitness:
            self.solution = new_solution
            self.fitness = new_fitness
            self.trial = 0
            return True
        else:
            self.trial += 1
            return False



