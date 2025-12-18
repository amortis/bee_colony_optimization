import math
from bees.bee import Bee
import random

# Импортируем оптимизированные функции для TSP
try:
    from tsp_optimizations import local_search_2opt, two_opt_swap, calculate_distance
    TSP_OPTIMIZATIONS_AVAILABLE = True
    
    def compute_fitness_from_distance_matrix(solution, distance_matrix):
        """ Вычисляет фитнес (1/расстояние) на основе матрицы расстояний. """
        if distance_matrix is None:
            return 0.0
        return 1.0 / calculate_distance(solution, distance_matrix)
except ImportError:
    TSP_OPTIMIZATIONS_AVAILABLE = False
    def compute_fitness_from_distance_matrix(solution, distance_matrix):
        if distance_matrix is None:
            return 0.0
        total_distance = 0
        n = len(solution)
        for i in range(n):
            city1 = solution[i]
            city2 = solution[(i + 1) % n]
            total_distance += distance_matrix[city1, city2]
        return 1.0 / total_distance if total_distance > 0 else 0.0

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

    # --- АСИНХРОННАЯ ВЕРСИЯ (ИЗМЕНЕНО) ---
    def explore_async(self, other_solutions, distance_matrix):
        """
        Асинхронная версия explore, возвращает (is_improved, new_solution, new_fitness).
        """
        self.distance_matrix = distance_matrix
        if not other_solutions:
            return False, self.solution, self.fitness
        probabilities = self._calculate_selection_probabilities_static(other_solutions, distance_matrix)
        selected_solution = self._select_solution_based_on_probability_static(other_solutions, probabilities, distance_matrix)
        new_solution = self._generate_new_solution(selected_solution)
        new_fitness = compute_fitness_from_distance_matrix(new_solution, distance_matrix)
        if new_fitness > self.fitness:
            return True, new_solution, new_fitness
        else:
            return False, self.solution, self.fitness

    # --- СТАТИЧЕСКАЯ ОБЕРТКА ДЛЯ ProcessPoolExecutor ---
    @staticmethod
    def explore_async_static_wrapper(args):
        initial_solution, employed_solutions, distance_matrix = args
        
        # Создаём временный объект OnlookerBee для доступа к его методам
        temp_onlooker = OnlookerBee(initial_solution, lambda s: 1/calculate_distance(s, distance_matrix) if TSP_OPTIMIZATIONS_AVAILABLE else 0, distance_matrix=distance_matrix)
        
        if not employed_solutions:
            return temp_onlooker.solution, temp_onlooker.fitness

        probabilities = OnlookerBee._calculate_selection_probabilities_static(employed_solutions, distance_matrix)
        selected_solution = OnlookerBee._select_solution_based_on_probability_static(employed_solutions, probabilities, distance_matrix)
        new_solution = temp_onlooker._generate_new_solution(selected_solution)
        new_fitness = compute_fitness_from_distance_matrix(new_solution, distance_matrix)

        if new_fitness > temp_onlooker.fitness:
            return new_solution, new_fitness
        else:
            return temp_onlooker.solution, temp_onlooker.fitness

    # --- СТАТИЧЕСКИЕ МЕТОДЫ (ИЗМЕНЕНО: принимают distance_matrix) ---
    @staticmethod
    def _calculate_selection_probabilities_static(solutions, distance_matrix):
        fitness_values = [compute_fitness_from_distance_matrix(sol, distance_matrix) for sol in solutions]
        max_fitness = max(fitness_values) if fitness_values else 1
        exp_values = [math.exp((f - max_fitness) * 10) for f in fitness_values]
        total = sum(exp_values)
        if total > 0:
            return [exp / total for exp in exp_values]
        else:
            return [1 / len(solutions)] * len(solutions)

    @staticmethod
    def _select_solution_based_on_probability_static(solutions, probabilities, distance_matrix):
        candidates = random.choices(solutions, weights=probabilities, k=3)
        return max(candidates, key=lambda x: compute_fitness_from_distance_matrix(x, distance_matrix))

    # --- НЕСТАТИЧЕСКИЕ МЕТОДЫ ---
    def _calculate_selection_probabilities(self, solutions):
        return self._calculate_selection_probabilities_static(solutions, self.distance_matrix)

    def _select_solution_based_on_probability(self, solutions, probabilities):
        return self._select_solution_based_on_probability_static(solutions, probabilities, self.distance_matrix)

    # --- ОСТАЛЬНЫЕ МЕТОДЫ ОСТАЮТСЯ ТЕМИ ЖЕ, НО БЕЗ self.fitness_function ---
    def _generate_new_solution(self, base_solution):
        """Генерирует модифицированное решение на основе базового.

        Использует несколько типов мутаций:
        - inversion, swap, shift (как было)
        - two_opt — локальное улучшение
        - double_bridge — сильная перестройка при больших trial.
        """
        new_solution = base_solution.copy()

        # Если наблюдатель давно не улучшался — пробуем сильную мутацию
        if self.trial > 20:
            mutated_solution = self._double_bridge_move(new_solution)
            if self.distance_matrix is not None and TSP_OPTIMIZATIONS_AVAILABLE:
                final_solution, _ = local_search_2opt(mutated_solution, self.distance_matrix)
                return final_solution
            return mutated_solution

        mutation_type = random.choice(["inversion", "swap", "shift"])

        if mutation_type == "inversion":
            i, j = sorted(random.sample(range(len(new_solution)), 2))
            new_solution[i:j + 1] = reversed(new_solution[i:j + 1])
        elif mutation_type == "swap":
            i, j = random.sample(range(len(new_solution)), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        elif mutation_type == "shift":
            city = new_solution.pop(random.randint(0, len(new_solution) - 1))
            new_solution.insert(random.randint(0, len(new_solution)), city)
        
        if self.distance_matrix is not None and TSP_OPTIMIZATIONS_AVAILABLE:
            final_solution, _ = local_search_2opt(new_solution, self.distance_matrix)
            return final_solution
        return new_solution

    def _two_opt_move(self, solution):
        """Один шаг 2-opt для маршрута."""
        if TSP_OPTIMIZATIONS_AVAILABLE:
            size = len(solution)
            i, j = sorted(random.sample(range(size), 2))
            return two_opt_swap(solution, i, j)
        else:
            size = len(solution)
            i, j = sorted(random.sample(range(size), 2))
            new_solution = solution.copy()
            new_solution[i:j + 1] = reversed(new_solution[i:j + 1])
            return new_solution

    def _double_bridge_move(self, solution):
        """Double-bridge мутация — агрессивная перестройка тура."""
        n = len(solution)
        if n < 8:
            if TSP_OPTIMIZATIONS_AVAILABLE:
                i, j = sorted(random.sample(range(n), 2))
                return two_opt_swap(solution, i, j)
            else:
                return self._two_opt_move(solution)

        a, b, c, d = sorted(random.sample(range(1, n - 1), 4))
        p1 = solution[:a]
        p2 = solution[a:b]
        p3 = solution[b:c]
        p4 = solution[c:d]
        p5 = solution[d:]
        return p1 + p3 + p2 + p4 + p5

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
