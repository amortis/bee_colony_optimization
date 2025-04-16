from bees.bee import Bee
import random
import numpy as np
from distane_matrix import DISTANCE_MATRIX


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

        # 3. Применяем локальный поиск 2-opt
        new_solution = self.two_opt(new_solution)

        # 4. Жадный выбор
        new_fitness = self.fitness_function(new_solution)
        if new_fitness > self.fitness:
            self.solution = new_solution
            self.fitness = new_fitness
            self.trial = 0
            return True
        else:
            self.trial += 1
            return False

    def two_opt(self, solution):
        """
        Применяет локальный поиск 2-opt к решению.
        """
        best_solution = solution.copy()
        best_fitness = self.fitness_function(best_solution)
        improved = True

        while improved:
            improved = False
            for i in range(len(solution) - 1):
                for j in range(i + 2, len(solution)):
                    # Создаем новое решение с перевернутым сегментом
                    new_solution = solution.copy()
                    new_solution[i:j] = solution[i:j][::-1]
                    
                    # Проверяем улучшение
                    new_fitness = self.fitness_function(new_solution)
                    if new_fitness > best_fitness:
                        best_solution = new_solution
                        best_fitness = new_fitness
                        improved = True
                        break
                if improved:
                    break
            solution = best_solution

        return best_solution

    def generate_new_solution(self, partner_solution):
        """
        Генерация нового решения с использованием различных операторов кроссовера.
        """
        # Выбираем оператор кроссовера с вероятностями
        crossover_type = random.choices(
            ['pmx', 'ox', 'cx', 'simple'],
            weights=[0.4, 0.3, 0.2, 0.1]
        )[0]

        if crossover_type == 'pmx':
            return self._pmx_crossover(partner_solution)
        elif crossover_type == 'ox':
            return self._ox_crossover(partner_solution)
        elif crossover_type == 'cx':
            return self._cx_crossover(partner_solution)
        else:
            return self._simple_crossover(partner_solution)

    def _pmx_crossover(self, partner_solution):
        """Частично отображенный кроссовер (PMX)"""
        size = len(self.solution)
        # Выбираем две точки кроссовера
        point1 = random.randint(0, size - 2)
        point2 = random.randint(point1 + 1, size - 1)
        
        # Создаем потомка
        child = [-1] * size
        
        # Копируем сегмент из первого родителя
        child[point1:point2] = self.solution[point1:point2]
        
        # Создаем отображение
        mapping = {}
        for i in range(point1, point2):
            mapping[self.solution[i]] = partner_solution[i]
        
        # Заполняем остальные позиции
        for i in range(size):
            if i < point1 or i >= point2:
                value = partner_solution[i]
                while value in child:
                    value = mapping.get(value, value)
                child[i] = value
        
        return child

    def _ox_crossover(self, partner_solution):
        """Упорядоченный кроссовер (OX)"""
        size = len(self.solution)
        # Выбираем две точки кроссовера
        point1 = random.randint(0, size - 2)
        point2 = random.randint(point1 + 1, size - 1)
        
        # Создаем потомка
        child = [-1] * size
        
        # Копируем сегмент из первого родителя
        child[point1:point2] = self.solution[point1:point2]
        
        # Заполняем остальные позиции из второго родителя
        ptr = 0
        for i in range(size):
            if i < point1 or i >= point2:
                while partner_solution[ptr] in child:
                    ptr += 1
                child[i] = partner_solution[ptr]
                ptr += 1
        
        return child

    def _cx_crossover(self, partner_solution):
        """Циклический кроссовер (CX)"""
        size = len(self.solution)
        child = [-1] * size
        visited = set()
        cycle_start = 0
        
        while len(visited) < size:
            if cycle_start not in visited:
                current = cycle_start
                cycle = []
                
                while current not in cycle:
                    cycle.append(current)
                    current = self.solution.index(partner_solution[current])
                
                # Копируем цикл из первого родителя
                for i in cycle:
                    child[i] = self.solution[i]
                    visited.add(i)
                
                # Ищем следующую непосещенную позицию
                cycle_start = next((i for i in range(size) if i not in visited), size)
        
        return child

    def _simple_crossover(self, partner_solution):
        """Простой кроссовер (старая версия)"""
        size = len(self.solution)
        new_solution = [-1] * size

        # Выбираем случайный отрезок (минимум 2 города)
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)

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
