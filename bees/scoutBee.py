from bees.bee import Bee
import random


class ScoutBee(Bee):
    """
    Класс, представляющий пчелу-разведчика. Пчелы-разведчики ищут новые случайные решения.
    """

    def __init__(self, solution, fitness_function, lb, ub):
        """
        Инициализация пчелы-разведчика.

        :param solution: Текущее решение пчелы.
        :param fitness_function: Функция, которая оценивает качество решения.
        :param lb: Нижняя граница пространства решений.
        :param ub: Верхняя граница пространства решений.
        """
        super().__init__(solution, fitness_function)
        self.lb = lb
        self.ub = ub

    def explore(self, other_solutions):
        """
        Пчела-разведчик генерирует новое случайное решение в пределах заданных границ.

        :param other_solutions: Список других решений (не используется в ScoutBee).
        :return: Возвращает True, если найдено лучшее решение, иначе False.
        """
        # Генерируем новое случайное решение
        new_solution = self.generate_random_solution(self.lb, self.ub)

        # Обновляем текущее решение, если новое лучше
        return self.update_solution(new_solution)

    def generate_random_solution(self, lb, ub):
        """
        Генерирует улучшенное случайное решение, комбинируя жадный алгоритм и случайные перестановки.
        """
        n = ub - lb + 1
        # Создаем базовое решение жадным алгоритмом
        solution = []
        current_city = random.randint(lb, ub)
        solution.append(current_city)
        remaining_cities = set(range(lb, ub + 1)) - {current_city}
        
        while remaining_cities:
            # С вероятностью 0.7 выбираем ближайший город, иначе случайный
            if random.random() < 0.7:
                next_city = min(remaining_cities, 
                              key=lambda city: DISTANCE_MATRIX[current_city][city])
            else:
                next_city = random.choice(list(remaining_cities))
            
            solution.append(next_city)
            remaining_cities.remove(next_city)
            current_city = next_city
        
        # Применяем несколько случайных перестановок
        for _ in range(3):
            i, j = random.sample(range(n), 2)
            solution[i], solution[j] = solution[j], solution[i]
        
        return solution
