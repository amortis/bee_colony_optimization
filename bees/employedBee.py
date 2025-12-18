from bees.bee import Bee
import random

# Импортируем оптимизированные функции для TSP
try:
    from tsp_optimizations import local_search_2opt, two_opt_swap, calculate_distance
    TSP_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    TSP_OPTIMIZATIONS_AVAILABLE = False

# --- ТОП-УРОВНЕВАЯ ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ ФИТНЕСА (для асинхронных операций) ---
def compute_fitness_from_distance_matrix(solution, distance_matrix):
    """ Вычисляет фитнес (1/расстояние) на основе матрицы расстояний. """
    if distance_matrix is None:
        return 0.0
    if TSP_OPTIMIZATIONS_AVAILABLE:
        return 1.0 / calculate_distance(solution, distance_matrix)
    else:
        # Fallback если модуль не доступен
        total_distance = 0
        n = len(solution)
        for i in range(n):
            city1 = solution[i]
            city2 = solution[(i + 1) % n]
            total_distance += distance_matrix[city1, city2]
        return 1.0 / total_distance if total_distance > 0 else 0.0


class EmployedBee(Bee):
    """
        Класс, представляющий рабочую пчелу. Рабочие пчелы отвечают за улучшение текущих решений.
    """
    def explore(self, other_solutions): # type: ignore
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
        Генерирует новое решение, применяя кроссовер и затем 2-opt.
        """
        current_solution = self.solution.copy()
        
        if self.trial > 20:
            mutated_solution = self._double_bridge_move(current_solution)
            if self.distance_matrix is not None and TSP_OPTIMIZATIONS_AVAILABLE:
                final_solution, _ = local_search_2opt(mutated_solution, self.distance_matrix)
                return final_solution
            return mutated_solution

        crossover_solution = self._ox_crossover(partner_solution)
        
        if self.distance_matrix is not None and TSP_OPTIMIZATIONS_AVAILABLE:
            final_solution, _ = local_search_2opt(crossover_solution, self.distance_matrix)
            return final_solution
        
        return crossover_solution

    def _ox_crossover(self, partner_solution):
        """OX-кроссовер (Order Crossover) между текущим и партнёрским решениями."""
        size = len(self.solution)
        new_solution = [-1] * size

        # Выбираем случайный отрезок (минимум 2 города)
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, min(start + size // 2, size))

        # Копируем сегмент из текущего решения
        new_solution[start:end] = self.solution[start:end]

        # Используем множество для быстрого поиска уже использованных городов
        used = set(self.solution[start:end])

        # Заполняем остальное из партнёрского решения (порядок сохранен)
        ptr = 0
        for i in range(size):
            if new_solution[i] == -1:
                while partner_solution[ptr] in used:
                    ptr += 1
                new_solution[i] = partner_solution[ptr]
                used.add(partner_solution[ptr])

        return new_solution

    def _two_opt_move(self, solution):
        """Один шаг 2-opt: разворот случайного подотрезка."""
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
        """
        Double-bridge мутация — сильная перестройка маршрута.
        Хорошо подходит для выхода из локальных минимумов.
        """
        n = len(solution)
        if n < 8:
            # Для маленьких туров достаточно 2-opt
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

    # --- АСИНХРОННАЯ ВЕРСИЯ (ИЗМЕНЕНО) ---
    def explore_async(self, other_solutions, distance_matrix):
        """
        Асинхронная версия explore, возвращает (is_improved, new_solution, new_fitness).
        """
        # ВАЖНО: Мы не можем использовать self.distance_matrix, так как объект EmployedBee
        # не сериализуется полностью при передаче в ProcessPoolExecutor.
        # Мы должны использовать переданный distance_matrix.
        
        # Создаем временный объект EmployedBee для использования его методов
        # (это не тот же объект, что в основном процессе, но он имеет те же методы)
        temp_bee = EmployedBee(self.solution, self.fitness_function, initial_fitness=self.fitness, distance_matrix=distance_matrix)
        
        # Выбираем партнера (нужно передать только решения, а не объекты пчел)
        partner_solutions = [bee.solution for bee in other_solutions if bee != self]
        if not partner_solutions:
            return False, self.solution, self.fitness
            
        partner_solution = random.choice(partner_solutions)
        
        # Генерируем новое решение с помощью generate_new_solution
        new_solution = temp_bee.generate_new_solution(partner_solution)
        
        # Вычисляем фитнес с помощью top-level функции
        new_fitness = compute_fitness_from_distance_matrix(new_solution, distance_matrix)

        if new_fitness > self.fitness:
            return True, new_solution, new_fitness
        else:
            return False, self.solution, self.fitness
