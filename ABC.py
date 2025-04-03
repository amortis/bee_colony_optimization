import random
from bees import EmployedBee, OnlookerBee
import matplotlib.pyplot as plt


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

        # Лучшее решение
        self.best_solution = None
        self.best_fitness = float('-inf')

        # Для визуализации
        self.history_on_looker_phase = []
        self.global_history = []

        # Для вычисления итераций без улучшения
        self.patience = 50  # Максимальное число итераций без улучшений
        self.wait = 0  # Счетчик итераций без улучшений
        self.best_iteration = 0  # Итерация, когда было найдено лучшее решение

    def _initialize_population(self) -> None:
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
            if not improved:
                onlooker.trial += 1  # Увеличиваем счетчик при неудаче

            # Добавляем пчелу в массив новых решений
            self.onlooker_bees.append(onlooker)

        self._upgrade_solutions()

    def _upgrade_solutions(self) -> None:
        """
        Обновляет лучшее решение после этапа пчел наблюдателей
        """
        for bee in self.onlooker_bees:
            self.history_on_looker_phase.append(self.best_fitness)
            if bee.fitness > self.best_fitness:
                self.best_solution = bee.solution
                self.best_fitness = bee.fitness

    def visualisation_on_looker_phase(self) -> None:
        plt.plot(self.history_on_looker_phase)
        plt.title("Сходимость алгоритма")
        plt.xlabel("Итерация")
        plt.ylabel("Фитнес")
        plt.show()

    def scout_bee_phase(self):
        """
        Фаза разведчиков: заменяет решения, которые не улучшались дольше limit итераций
        """
        for i, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                # Генерируем совершенно новое случайное решение
                new_solution = self._generate_random_solution()
                new_fitness = self.fitness_function(new_solution)

                # Заменяем "застрявшее" решение
                self.employed_bees[i].solution = new_solution
                self.employed_bees[i].fitness = new_fitness
                self.employed_bees[i].trial = 0

                # Проверяем, не нашли ли мы новое лучшее решение
                if new_fitness > self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness

    def _generate_random_solution(self):
        """Генерирует полностью случайное решение для задачи коммивояжера"""
        solution = list(range(self.lb, self.ub + 1))
        random.shuffle(solution)
        return solution

    def run_algorithm(self, max_iterations):
        """
        Полный цикл выполнения алгоритма
        """
        self._initialize_population()

        for iteration in range(max_iterations):
            old_best = self.best_fitness
            # Фаза рабочих пчел
            self.employed_bee_phase()

            # Фаза пчел-наблюдателей
            self.onlooker_bee_phase()

            # Фаза разведчиков
            self.scout_bee_phase()

            self.global_history.append(self.best_fitness)
            # Проверка улучшения
            if self.best_fitness > old_best:
                self.wait = 0
                self.best_iteration = iteration
            else:
                self.wait += 1

            # Ранняя остановка
            if self.wait >= self.patience:
                print(f"\nEarly stopping at iteration {iteration}")
                print(f"No improvement for {self.patience} iterations")
                break

            # Логирование (можно настроить по желанию)
            if iteration % 3 == 0:
                print(f"Iteration {iteration}: Best distance = {1 / self.best_fitness:.2f}")

        self.plot_convergence(self.global_history)
        return self.best_solution, 1 / self.best_fitness

    def plot_convergence(self, history):
        """
        Визуализация
        """
        plt.plot(history)
        plt.title("Convergence History")
        plt.xlabel("Iteration")
        plt.ylabel("Best Distance")
        plt.show()