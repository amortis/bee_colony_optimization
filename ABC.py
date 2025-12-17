import random
import time
from datetime import timedelta


import numpy as np

from bees import EmployedBee, OnlookerBee
import matplotlib.pyplot as plt

import concurrent.futures
import multiprocessing as mp


# --- НОВАЯ ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ ФИТНЕСА В ОТДЕЛЬНОМ ПРОЦЕССЕ ---
# Эта функция должна быть на уровне модуля, чтобы быть pickle-совместимой
def compute_fitness_task(args):
    solution, fitness_func = args
    return fitness_func(solution)

class ABCAlgorithm:
    def __init__(self, fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, limit, patience, optimal_length, num_workers=None, seed=42):
        """
        Инициализация алгоритма.

        :param fitness_function: Функция, которая оценивает качество решения.
        :param lb: Нижняя граница пространства решений.
        :param ub: Верхняя граница пространства решений.
        :param num_employed_bees: Количество рабочих пчел.
        :param num_onlooker_bees: Количество пчел-наблюдателей.
        :param limit: Максимальное количество неудачных попыток улучшения решения.
        """
        # проверки
        assert ub > lb, "Верхняя граница должна быть больше нижней"
        assert num_employed_bees > 0, "Должна быть хотя бы одна рабочая пчела"
        assert num_onlooker_bees > 0, "Должна быть хотя бы одна пчела-наблюдатель"
        assert limit > 0, "Лимит неудач должен быть положительным"
        # Сиды
        # Фиксируем случайность
        # random.seed(seed)
        # np.random.seed(seed)
        # self.seed = seed

        # --- НОВОЕ ---
        self.num_workers = num_workers or min(32, mp.cpu_count()) # По умолчанию используем количество ядер
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
        # --- КОНЕЦ НОВОГО ---


        self.fitness_function = fitness_function
        self.lb = lb
        self.ub = ub
        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.limit = limit
        self.optimal_length = optimal_length

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

    # --- ИЗМЕНЁННЫЙ МЕТОД ЗАВЕРШЕНИЯ ---
    def run_algorithm(self, max_iterations):
        """
        Полный цикл выполнения алгоритма
        """
        try:
            self._initialize_population()
            self.start_time = time.time()

            for iteration in range(max_iterations):
                old_best = self.best_fitness
                self.employed_bee_phase()
                self.onlooker_bee_phase()
                self.scout_bee_phase()
                self.global_history.append(1/self.best_fitness - self.optimal_length)

                if self.best_fitness > old_best:
                    self.wait = 0
                    self.best_iteration = iteration
                else:
                    self.wait += 1

                if self.wait >= self.patience:
                    print(f"\nEarly stopping at iteration {iteration}")
                    print(f"No improvement for {self.patience} iterations")
                    break

                if iteration % 20 == 0:
                    elapsed = self.get_formatted_time()
                    print(f"Iteration {iteration}. Time: {elapsed}. Best distance = {1 / self.best_fitness:.2f}")

            self.plot_convergence(self.global_history)
            return self.best_solution, self.best_fitness
        finally:
            # --- ВАЖНО: Закрытие пула процессов ---
            self.executor.shutdown(wait=True) # Добавляем shutdown
        

    def get_formatted_time(self, seconds=None):
        """Форматирует время в читаемый вид (HH:MM:SS)"""
        if seconds is None:
            seconds = self.get_elapsed_time()
        return str(timedelta(seconds=seconds)).split(".")[0]

    def get_elapsed_time(self):
        """Возвращает время выполнения в секундах"""
        return time.time() - self.start_time # type: ignore

    # --- ИЗМЕНЁННЫЙ МЕТОД _initialize_population ---
    def _initialize_population(self) -> None:
        """
        Инициализация начальной популяции пчел с многопоточностью.
        """
        # Генерация начальных решений
        initial_solutions = []
        for _ in range(self.num_employed_bees):
            solution = random.sample(range(self.lb, self.ub + 1), self.ub - self.lb + 1)
            initial_solutions.append(solution)

        # Подготовка задач для пула
        tasks = [(sol, self.fitness_function) for sol in initial_solutions]

        # Вычисление фитнеса в пуле
        fitness_results = list(self.executor.map(compute_fitness_task, tasks))

        # Создание пчёл
        for i, solution in enumerate(initial_solutions):
            employed_bee = EmployedBee(solution, self.fitness_function, initial_fitness=fitness_results[i])
            self.employed_bees.append(employed_bee)

            # Обновление лучшего решения
            if employed_bee.fitness > self.best_fitness:
                self.best_solution = employed_bee.solution
                self.best_fitness = employed_bee.fitness

    # --- ИЗМЕНЁННАЯ ФАЗА РАБОЧИХ ПЧЁЛ ---
    def employed_bee_phase(self) -> None:
        """
        Фаза занятых пчел: рабочие пчелы улучшают свои решения с многопоточностью.
        """
        futures = []
        for bee in self.employed_bees:
            # Подготовка задачи для explore
            # explore теперь возвращает (is_improved, new_solution, new_fitness)
            future = self.executor.submit(bee.explore_async, self.employed_bees, self.fitness_function)
            futures.append(future)

        # Сбор результатов
        results = [future.result() for future in futures]

        for i, (is_improved, new_solution, new_fitness) in enumerate(results):
            bee = self.employed_bees[i]
            if is_improved:
                bee.solution = new_solution
                bee.fitness = new_fitness
                bee.trial = 0
                if new_fitness > self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness
            else:
                bee.trial += 1
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

    # --- ИЗМЕНЁННАЯ ФАЗА НАБЛЮДАТЕЛЕЙ ---
    def onlooker_bee_phase(self) -> None:
        """
        Фаза пчел-наблюдателей с многопоточностью.
        """
        if not self.employed_bees:
            return

        self.onlooker_bees.clear()
        employed_solutions = [bee.solution for bee in self.employed_bees]

        # Создаём задачи для onlooker bees
        onlooker_tasks = []
        for _ in range(self.num_onlooker_bees):
            # Создаем пчелу-наблюдателя с случайным начальным решением
            initial_solution = random.choice(employed_solutions)
            onlooker_tasks.append((initial_solution, employed_solutions, self.fitness_function))

        # Запускаем задачи асинхронно
        futures = [self.executor.submit(OnlookerBee.explore_async_static, task) for task in onlooker_tasks]

        # Сбор результатов
        results = [future.result() for future in futures]

        # Создание объектов OnlookerBee с обновлёнными данными
        self.onlooker_bees = []
        for solution, fitness in results:
            onlooker_bee = OnlookerBee(solution, self.fitness_function, initial_fitness=fitness)
            self.onlooker_bees.append(onlooker_bee)

        self._upgrade_solutions()
        """
        Фаза пчел-наблюдателей. Выбирает решения на основе вероятности и пытается их улучшить.
        """
        if not self.employed_bees:
            return
        self.onlooker_bees.clear() # Очищаем список от пчёл прошлой итерации
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
        # 1. Найдём лучшую пчелу-наблюдателя
        best_onlooker = max(self.onlooker_bees, key=lambda b: b.fitness)
        
        # 2. Найдём худшую рабочую пчелу
        worst_employed_idx = min(range(len(self.employed_bees)), 
                                key=lambda i: self.employed_bees[i].fitness)
        
        # 3. Если наблюдатель лучше худшей рабочей — заменяем
        if best_onlooker.fitness > self.employed_bees[worst_employed_idx].fitness:
            self.employed_bees[worst_employed_idx].solution = best_onlooker.solution.copy()
            self.employed_bees[worst_employed_idx].fitness = best_onlooker.fitness
            self.employed_bees[worst_employed_idx].trial = 0
        
        # 4. Обновляем глобальное лучшее решение
        for bee in self.onlooker_bees:
            if bee.fitness > self.best_fitness:
                self.best_solution = bee.solution.copy()
                self.best_fitness = bee.fitness

    def visualisation_on_looker_phase(self) -> None:
        plt.plot(self.history_on_looker_phase)
        plt.title("Сходимость алгоритма")
        plt.xlabel("Итерация")
        plt.ylabel("Фитнес")
        plt.show()

    # --- ИЗМЕНЁННАЯ ФАЗА РАЗВЕДЧИКОВ ---
    def scout_bee_phase(self):
        """
        Фаза разведчиков: заменяет решения, которые не улучшались дольше limit итераций, с многопоточностью.
        """
        scouts_to_update = []
        for i, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                scouts_to_update.append(i)

        if not scouts_to_update:
             return # Нечего обновлять

        # Генерируем новые решения
        new_solutions = [self._generate_random_solution() for _ in scouts_to_update]

        # Подготовка задач для вычисления фитнеса
        tasks = [(sol, self.fitness_function) for sol in new_solutions]

        # Вычисление фитнеса в пуле
        new_fitnesses = list(self.executor.map(compute_fitness_task, tasks))

        # Обновление пчёл
        for idx, sol, fit in zip(scouts_to_update, new_solutions, new_fitnesses):
            bee = self.employed_bees[idx]
            bee.solution = sol
            bee.fitness = fit
            bee.trial = 0

            # Проверяем, не нашли ли мы новое лучшее решение
            if fit > self.best_fitness:
                self.best_solution = sol.copy()
                self.best_fitness = fit
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
