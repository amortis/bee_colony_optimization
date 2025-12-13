import math
from bees.bee import Bee
import random

class OnlookerBee(Bee):
    """
    Класс, представляющий пчелу-наблюдателя. Пчелы-наблюдатели выбирают решения на основе их качества (фитнеса).
    """

    def explore(self, other_solutions):
        """
        Синхронная версия explore.
        Использует self.fitness_function.
        """
        if not other_solutions:
            return False

        # Вызов нестатических методов, которые используют self.fitness_function
        probabilities = self._calculate_selection_probabilities(other_solutions)
        selected_solution = self._select_solution_based_on_probability(other_solutions, probabilities)
        new_solution = self._generate_new_solution(selected_solution)
        return self._greedy_selection(new_solution)

    # --- РЕАЛИЗАЦИЯ АБСТРАКТНОГО МЕТОДА explore_async ---
    def explore_async(self, other_solutions, fitness_function):
        """
        Асинхронная версия explore, возвращает (is_improved, new_solution, new_fitness).
        Реализует синхронную логику для соответствия абстрактному методу, но принимает fitness_function.
        """
        if not other_solutions:
            return False, self.solution, self.fitness # Возвращаем старые значения

        # Вызов статических методов, передавая fitness_function
        probabilities = self._calculate_selection_probabilities_static(other_solutions, fitness_function)
        selected_solution = self._select_solution_based_on_probability_static(other_solutions, probabilities, fitness_function)
        new_solution = self._generate_new_solution(selected_solution)

        new_fitness = fitness_function(new_solution) # Вычисляем фитнес с переданной функцией

        # Жадный выбор
        if new_fitness > self.fitness:
            return True, new_solution, new_fitness
        else:
            return False, self.solution, self.fitness # Возвращаем исходное, если не улучшено

    # --- НОВАЯ СТАТИЧЕСКАЯ АСИНХРОННАЯ ВЕРСИЯ (для ProcessPoolExecutor) ---
    @staticmethod
    def explore_async_static(args):
        """
        Статический метод для асинхронного выполнения в отдельном процессе.
        Args: (initial_solution, employed_solutions, fitness_function)
        Returns: (new_solution, new_fitness)
        """
        initial_solution, employed_solutions, fitness_function = args
        # Создаём временный объект OnlookerBee с переданной функцией и решением
        # Мы не используем его внутренний фитнес, а вычисляем заново
        onlooker = OnlookerBee(initial_solution, fitness_function)

        if not employed_solutions:
            return onlooker.solution, onlooker.fitness

        # Вызов внутренних статических методов через класс OnlookerBee
        probabilities = OnlookerBee._calculate_selection_probabilities_static(employed_solutions, fitness_function)
        selected_solution = OnlookerBee._select_solution_based_on_probability_static(employed_solutions, probabilities, fitness_function)
        new_solution = onlooker._generate_new_solution(selected_solution)

        new_fitness = fitness_function(new_solution) # Вычисляем фитнес

        if new_fitness > onlooker.fitness:
            return new_solution, new_fitness
        else:
            return onlooker.solution, onlooker.fitness # Возвращаем исходное, если не улучшено

    # --- СТАТИЧЕСКИЕ МЕТОДЫ (для асинхронной версии) ---
    @staticmethod
    def _calculate_selection_probabilities_static(solutions, fitness_function):
        """Вычисляет вероятности выбора для каждого решения"""
        fitness_values = [fitness_function(sol) for sol in solutions]
        max_fitness = max(fitness_values) if fitness_values else 1
        exp_values = [math.exp((f - max_fitness) * 10) for f in fitness_values]
        total = sum(exp_values)
        if total > 0:
            return [exp / total for exp in exp_values]
        else:
            return [1 / len(solutions)] * len(solutions)

    @staticmethod
    def _select_solution_based_on_probability_static(solutions, probabilities, fitness_function):
        """Выбирает решение для исследования на основе вероятностей"""
        candidates = random.choices(solutions, weights=probabilities, k=3)
        return max(candidates, key=lambda x: fitness_function(x))

    # --- НЕСТАТИЧЕСКИЕ МЕТОДЫ (для синхронной версии explore) ---
    def _calculate_selection_probabilities(self, solutions):
        """Вычисляет вероятности выбора для каждого решения, используя self.fitness_function."""
        return self._calculate_selection_probabilities_static(solutions, self.fitness_function)

    def _select_solution_based_on_probability(self, solutions, probabilities):
        """Выбирает решение для исследования на основе вероятностей, используя self.fitness_function."""
        return self._select_solution_based_on_probability_static(solutions, probabilities, self.fitness_function)

    # --- ОСТАЛЬНЫЕ МЕТОДЫ ОСТАЮТСЯ ТЕМИ ЖЕ, НО БЕЗ self.fitness_function ---
    def _generate_new_solution(self, base_solution):
        """Генерирует модифицированное решение на основе базового"""
        new_solution = base_solution.copy()
        mutation_type = random.choice(["inversion", "swap", "shift"])

        if mutation_type == "inversion":
            i, j = sorted(random.sample(range(len(new_solution)), 2))
            new_solution[i:j + 1] = reversed(new_solution[i:j + 1])
        elif mutation_type == "swap":
            i, j = random.sample(range(len(new_solution)), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        else:
            city = new_solution.pop(random.randint(0, len(new_solution) - 1))
            new_solution.insert(random.randint(0, len(new_solution)), city)

        return new_solution

    def _greedy_selection(self, new_solution):
        """Применяет жадный выбор с обновлением состояния, используя self.fitness_function."""
        new_fitness = self.fitness_function(new_solution)

        if new_fitness > self.fitness:
            self.solution = new_solution
            self.fitness = new_fitness
            self.trial = 0
            return True
        else:
            self.trial += 1
            return False
