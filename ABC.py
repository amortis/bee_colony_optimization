import random
from bees import EmployedBee, OnlookerBee


class ABCAlgorithm:
    def __init__(self, fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, limit):
        """
        Инициализация алгоритма.

        :param fitness_function: Функция, которая оценивает качество решения.
        :param lb: Нижняя граница пространства решений.
        :param ub: Верхняя граница пространства решений.
        :param num_employed_bees: Количество рабочих пчел.
        :param num_onlooker_bees: Количество пчел-наблюдателей.
        :param limit: Максимальное количество неудачных попыток улучшения решения.
        """
        self.fitness_function = fitness_function
        self.lb = lb
        self.ub = ub
        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.limit = limit

        # Инициализация популяции пчел
        self.employed_bees = []
        self.onlooker_bees = []
        self.scout_bees = []

        # Лучшее решение
        self.best_solution = None
        self.best_fitness = float('-inf')

    def initialize_population(self) -> None:
        """
        Инициализация начальной популяции пчел.
        """
        for _ in range(self.num_employed_bees):
            # Генерация случайного решения
            solution = random.sample(range(self.lb, self.ub + 1), self.ub - self.lb + 1)
            employed_bee = EmployedBee(solution, self.fitness_function)
            self.employed_bees.append(employed_bee)

            # Обновление лучшего решения
            if employed_bee.fitness > self.best_fitness:
                self.best_solution = employed_bee.solution
                self.best_fitness = employed_bee.fitness

    def employed_bee_phase(self) -> None:
        """
        Фаза занятых пчел: рабочие пчелы улучшают свои решения.
        Обновляет счетчик trial и проверяет на улучшение решения.
        """
        for bee in self.employed_bees:
            is_improved = bee.explore(self.employed_bees)  # Захватываем флаг
            if is_improved and bee.fitness > self.best_fitness:
                self.best_solution = bee.solution
                self.best_fitness = bee.fitness
            else:
                bee.trial += 1

    def onlooker_bee_phase(self) -> None:
        """
        Фаза пчел-наблюдателей. Выбирает решения на основе вероятности и пытается их улучшить.
        """
        if not self.employed_bees:
            return
        # 1. Подготовка списка решений и их фитнес-значений
        solutions = [bee.solution for bee in self.employed_bees]
        # Значения фитнеса хранятся в классах пчел

        # 2. Вычисление вероятностей выбора
        # Данный шаг уже предусмотрен в классе Bee

        # 3. Создаем временный список для новых решений
        # у нас уже есть для этого self.onlooker_bees

        # 4. Каждая пчела-наблюдатель выбирает и улучшает решение
        for bee_index in range(self.num_onlooker_bees):
            # Создаем пчелу-наблюдателя с случайным начальным решением
            onlooker = OnlookerBee(random.choice(solutions), self.fitness_function)

            # Пчела выбирает и улучшает решение
            improved = onlooker.explore(solutions)

            # Добавляем пчелу в массив новых решений
            self.onlooker_bees.append(onlooker)

        self._upgrade_solutions()

    def _upgrade_solutions(self) -> None:
        """
        Обновляет лучшее решение после этапа пчел наблюдателей
        """
        for bee in self.onlooker_bees:
            if bee.fitness > self.best_fitness:
                self.best_solution = bee.solution
                self.best_fitness = bee.fitness
