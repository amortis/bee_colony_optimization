from bees.bee import Bee
import random


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
        Генерирует новое решение на основе текущего и партнёрского решения.
        Включает несколько типов операторов:
        - OX-кроссовер (как было раньше)
        - локальный 2-opt
        - более сильная перестройка (double-bridge), когда пчела долго не улучшалась.
        """
        # Если пчела давно не улучшалась — применяем более сильную мутацию
        if self.trial > 20:
            return self._double_bridge_move(self.solution)

        mutation_type = random.choice(["ox_crossover", "two_opt"])

        if mutation_type == "two_opt":
            return self._two_opt_move(self.solution)
        else:
            return self._ox_crossover(partner_solution)

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
            return self._two_opt_move(solution)

        new_solution = solution.copy()
        # Выбираем 4 точки разреза
        a, b, c, d = sorted(random.sample(range(1, n - 1), 4))
        p1 = new_solution[:a]
        p2 = new_solution[a:b]
        p3 = new_solution[b:c]
        p4 = new_solution[c:d]
        p5 = new_solution[d:]

        # Переставляем блоки: p1 + p3 + p2 + p4 + p5
        return p1 + p3 + p2 + p4 + p5

    # --- НОВАЯ АСИНХРОННАЯ ВЕРСИЯ ---
    def explore_async(self, other_solutions, fitness_function): # type: ignore
        """
        Асинхронная версия explore, возвращает (is_improved, new_solution, new_fitness).
        """
        partner = random.choice([bee for bee in other_solutions if bee != self])
        new_solution = self.generate_new_solution(partner.solution)
        new_fitness = fitness_function(new_solution) # Используем переданную функцию

        if new_fitness > self.fitness:
            return True, new_solution, new_fitness
        else:
            return False, self.solution, self.fitness # Возвращаем старые значения, если не улучшено
