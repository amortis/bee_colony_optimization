import random
import time
from datetime import timedelta
from distane_matrix import OPTIMAL_LENGTH, DISTANCE_MATRIX

import numpy as np

from bees import EmployedBee, OnlookerBee
import matplotlib.pyplot as plt


class ABCAlgorithm:
    def __init__(self, fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, limit, patience, seed=42):
        """
        Инициализация алгоритма.

        :param fitness_function: Функция, которая оценивает качество решения.
        :param lb: Нижняя граница пространства решений.
        :param ub: Верхняя граница пространства решений.
        :param num_employed_bees: Количество рабочих пчел.
        :param num_onlooker_bees: Количество пчел-наблюдателей.
        :param limit: Максимальное количество неудачных попыток улучшения решения.
        """
        # Сиды
        # Фиксируем случайность
        # random.seed(seed)
        # np.random.seed(seed)
        # self.seed = seed


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
        self.patience = patience  # Максимальное число итераций без улучшений
        self.wait = 0  # Счетчик итераций без улучшений
        self.best_iteration = 0  # Итерация, когда было найдено лучшее решение

        # Отсчет времени
        self.start_time = None
        self.end_time = None

    def run_algorithm(self, max_iterations):
        """
        Полный цикл выполнения алгоритма
        """
        self._initialize_population()
        self.start_time = time.time()

        for iteration in range(max_iterations):
            old_best = self.best_fitness
            # Фаза рабочих пчел
            self.employed_bee_phase()

            # Фаза пчел-наблюдателей
            self.onlooker_bee_phase()

            # Фаза разведчиков
            self.scout_bee_phase()

            # Добавление данных для визуализации
            self.global_history.append(1/self.best_fitness - OPTIMAL_LENGTH)
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
            if iteration % 50 == 0:
                elapsed = self.get_formatted_time()
                print(f"Iteration {iteration}. Time: {elapsed}. Best distance = {1 / self.best_fitness:.2f}")

        #self.plot_convergence(self.global_history)
        return self.best_solution, self.best_fitness

    def get_formatted_time(self, seconds=None):
        """Форматирует время в читаемый вид (HH:MM:SS)"""
        if seconds is None:
            seconds = self.get_elapsed_time()
        return str(timedelta(seconds=seconds)).split(".")[0]

    def get_elapsed_time(self):
        """Возвращает время выполнения в секундах"""
        return time.time() - self.start_time

    def _initialize_population(self) -> None:
        """
        Инициализация начальной популяции пчел.
        """
        # Разделяем пчел на три группы для разных методов инициализации
        num_greedy = self.num_employed_bees // 3
        num_random = self.num_employed_bees // 3
        num_christofides = self.num_employed_bees - num_greedy - num_random

        # Инициализация жадным алгоритмом с разными стартовыми точками
        for i in range(num_greedy):
            solution = self._greedy_initial_solution(start_city=i)
            employed_bee = EmployedBee(solution, self.fitness_function)
            self.employed_bees.append(employed_bee)
            if employed_bee.fitness > self.best_fitness:
                self.best_solution = employed_bee.solution
                self.best_fitness = employed_bee.fitness

        # Инициализация случайными решениями
        for _ in range(num_random):
            solution = self._random_initial_solution()
            employed_bee = EmployedBee(solution, self.fitness_function)
            self.employed_bees.append(employed_bee)
            if employed_bee.fitness > self.best_fitness:
                self.best_solution = employed_bee.solution
                self.best_fitness = employed_bee.fitness

        # Инициализация алгоритмом Кристофидеса
        for _ in range(num_christofides):
            solution = self._christofides_initial_solution()
            employed_bee = EmployedBee(solution, self.fitness_function)
            self.employed_bees.append(employed_bee)
            if employed_bee.fitness > self.best_fitness:
                self.best_solution = employed_bee.solution
                self.best_fitness = employed_bee.fitness

    def _greedy_initial_solution(self, start_city=0):
        """
        Генерация начального решения с использованием жадного алгоритма (ближайший сосед).
        """
        cities = list(range(self.lb, self.ub + 1))
        solution = [cities.pop(start_city)]  # Начинаем с заданного города

        while cities:
            last_city = solution[-1]
            next_city = min(cities, key=lambda city: DISTANCE_MATRIX[last_city][city])
            solution.append(next_city)
            cities.remove(next_city)

        return solution

    def _random_initial_solution(self):
        """
        Генерация случайного начального решения.
        """
        cities = list(range(self.lb, self.ub + 1))
        random.shuffle(cities)
        return cities

    def _christofides_initial_solution(self):
        """
        Генерация начального решения с использованием алгоритма Кристофидеса.
        Это приближенный алгоритм, который гарантирует решение не хуже 1.5 * оптимального.
        """
        # 1. Построение минимального остовного дерева
        n = self.ub - self.lb + 1
        mst = self._prim_mst()
        
        # 2. Нахождение вершин нечетной степени
        odd_vertices = [i for i in range(n) if len([j for j in range(n) if mst[i][j]]) % 2 == 1]
        
        # 3. Построение минимального паросочетания на нечетных вершинах
        matching = self._min_weight_matching(odd_vertices)
        
        # 4. Объединение MST и паросочетания
        multigraph = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if mst[i][j] or matching[i][j]:
                    multigraph[i][j] = True
                    multigraph[j][i] = True
        
        # 5. Нахождение эйлерова цикла
        euler_tour = self._find_euler_tour(multigraph)
        
        # 6. Преобразование в гамильтонов цикл
        solution = []
        visited = set()
        for city in euler_tour:
            if city not in visited:
                solution.append(city + self.lb)
                visited.add(city)
        
        return solution

    def _prim_mst(self):
        """Алгоритм Прима для построения минимального остовного дерева"""
        n = self.ub - self.lb + 1
        mst = [[False] * n for _ in range(n)]
        key = [float('inf')] * n
        parent = [-1] * n
        key[0] = 0
        mst_set = [False] * n

        for _ in range(n):
            u = min((i for i in range(n) if not mst_set[i]), key=lambda x: key[x])
            mst_set[u] = True

            if parent[u] != -1:
                mst[parent[u]][u] = True
                mst[u][parent[u]] = True

            for v in range(n):
                if (DISTANCE_MATRIX[u + self.lb][v + self.lb] < key[v] and 
                    not mst_set[v]):
                    key[v] = DISTANCE_MATRIX[u + self.lb][v + self.lb]
                    parent[v] = u

        return mst

    def _min_weight_matching(self, odd_vertices):
        """Построение минимального паросочетания на нечетных вершинах"""
        n = self.ub - self.lb + 1
        matching = [[False] * n for _ in range(n)]
        visited = set()

        while len(visited) < len(odd_vertices):
            u = next(v for v in odd_vertices if v not in visited)
            visited.add(u)
            
            min_dist = float('inf')
            best_v = -1
            
            for v in odd_vertices:
                if v != u and v not in visited:
                    dist = DISTANCE_MATRIX[u + self.lb][v + self.lb]
                    if dist < min_dist:
                        min_dist = dist
                        best_v = v
            
            if best_v != -1:
                matching[u][best_v] = True
                matching[best_v][u] = True
                visited.add(best_v)

        return matching

    def _find_euler_tour(self, multigraph):
        """Нахождение эйлерова цикла в мультиграфе"""
        n = len(multigraph)
        tour = []
        stack = [0]
        
        while stack:
            u = stack[-1]
            for v in range(n):
                if multigraph[u][v]:
                    multigraph[u][v] = False
                    multigraph[v][u] = False
                    stack.append(v)
                    break
            else:
                tour.append(stack.pop())
        
        return tour[::-1]

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


    def plot_convergence(self, history):
        """
        Визуализация
        """
        plt.plot(history)
        plt.title("График сходимости")
        plt.xlabel("Итерация")
        plt.ylabel("Разница с оптимумом")
        plt.show()
