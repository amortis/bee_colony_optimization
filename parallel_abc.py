import random
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt

# Предполагается, что OPTIMAL_LENGTH определена в этом модуле или импортируется
# from distane_matrix import OPTIMAL_LENGTH
# Для примера, определим её как 0
OPTIMAL_LENGTH = 0

from bees.bee import EmployedBee, OnlookerBee

class ABCAlgorithm:
    def __init__(self, fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, limit, patience, seed=42, max_workers=None):
        assert ub > lb, "Верхняя граница должна быть больше нижней"
        assert num_employed_bees > 0, "Должна быть хотя бы одна рабочая пчела"
        assert num_onlooker_bees > 0, "Должна быть хотя бы одна пчела-наблюдатель"
        assert limit > 0, "Лимит неудач должен быть положительным"

        random.seed(seed)
        np.random.seed(seed)

        self.fitness_function = fitness_function
        self.lb = lb
        self.ub = ub
        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.limit = limit
        self.patience = patience
        self.wait = 0
        self.best_iteration = 0

        self.employed_bees = []
        self.onlooker_bees = []
        self.best_solution = None
        self.best_fitness = float('-inf')

        self.history_on_looker_phase = []
        self.global_history = []

        self.start_time = None
        self.end_time = None

        # Количество потоков для параллельных вычислений
        self.max_workers = max_workers or min(32, (num_employed_bees + num_onlooker_bees + 1) // 2)

    def run_algorithm(self, max_iterations):
        self._initialize_population()
        self.start_time = time.time()

        for iteration in range(max_iterations):
            old_best = self.best_fitness

            # Фаза рабочих пчел (многопоточная)
            self.employed_bee_phase()

            # Фаза пчел-наблюдателей (многопоточная)
            self.onlooker_bee_phase()

            # Фаза разведчиков
            self.scout_bee_phase()

            self.global_history.append(1/self.best_fitness - OPTIMAL_LENGTH)

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

    def get_formatted_time(self, seconds=None):
        if seconds is None:
            seconds = self.get_elapsed_time()
        return str(timedelta(seconds=seconds)).split(".")[0]

    def get_elapsed_time(self):
        return time.time() - self.start_time

    def _initialize_population(self):
        for _ in range(self.num_employed_bees):
            solution = random.sample(range(self.lb, self.ub + 1), self.ub - self.lb + 1)
            employed_bee = EmployedBee(solution, self.fitness_function)
            self.employed_bees.append(employed_bee)

            if employed_bee.fitness > self.best_fitness:
                self.best_solution = employed_bee.solution.copy() # Сохраняем копию
                self.best_fitness = employed_bee.fitness

    def employed_bee_phase(self):
        # Подготовка данных для задач
        bee_data = [(bee, self.employed_bees) for bee in self.employed_bees]

        updated_bees = []
        improvements = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Отправляем задачи в пул потоков
            future_to_index = {executor.submit(self._process_employed_bee_task, data): i for i, data in enumerate(bee_data)}

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    updated_bee, improved = future.result()
                    updated_bees.append((index, updated_bee))
                    improvements.append(improved)
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        # Обновляем популяцию и лучшее решение в основном потоке
        for index, updated_bee in updated_bees:
            self.employed_bees[index] = updated_bee
            if updated_bee.fitness > self.best_fitness:
                self.best_solution = updated_bee.solution.copy()
                self.best_fitness = updated_bee.fitness

    def _process_employed_bee_task(self, data):
        bee, other_solutions = data
        # Создаем копию пчелы для работы в потоке
        bee_copy = EmployedBee(bee.solution.copy(), bee.fitness_function)
        bee_copy.fitness = bee.fitness
        bee_copy.trial = bee.trial

        is_improved = bee_copy.explore([EmployedBee(b.solution.copy(), b.fitness_function) if b != bee else bee_copy for b in other_solutions])
        return bee_copy, is_improved

    def onlooker_bee_phase(self):
        if not self.employed_bees:
            return

        self.onlooker_bees.clear()
        solutions = [bee.solution for bee in self.employed_bees]

        # Подготовка данных для задач пчёл-наблюдателей
        onlooker_tasks = [solutions for _ in range(self.num_onlooker_bees)]

        updated_onlooker_bees = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Отправляем задачи в пул потоков
            future_to_index = {executor.submit(self._process_onlooker_bee_task, task): i for i, task in enumerate(onlooker_tasks)}

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    updated_onlooker = future.result()
                    updated_onlooker_bees.append(updated_onlooker)
                except Exception as exc:
                    print(f'Generated an exception for onlooker: {exc}')

        # Обновляем список пчёл-наблюдателей
        self.onlooker_bees = updated_onlooker_bees

        # Обновляем решения рабочих пчёл и лучшее решение
        self._upgrade_solutions()

    def _process_onlooker_bee_task(self, solutions):
        onlooker = OnlookerBee(random.choice(solutions), self.fitness_function)
        # Передаем оригинальные решения, так как explore не изменяет их напрямую
        improved = onlooker.explore(solutions)
        return onlooker

    def _upgrade_solutions(self):
        if not self.onlooker_bees:
            return

        best_onlooker = max(self.onlooker_bees, key=lambda b: b.fitness)
        worst_employed_idx = min(range(len(self.employed_bees)), key=lambda i: self.employed_bees[i].fitness)

        if best_onlooker.fitness > self.employed_bees[worst_employed_idx].fitness:
            self.employed_bees[worst_employed_idx].solution = best_onlooker.solution.copy()
            self.employed_bees[worst_employed_idx].fitness = best_onlooker.fitness
            self.employed_bees[worst_employed_idx].trial = 0

        for bee in self.onlooker_bees:
            if bee.fitness > self.best_fitness:
                self.best_solution = bee.solution.copy()
                self.best_fitness = bee.fitness

    def scout_bee_phase(self):
        for i, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                new_solution = self._generate_random_solution()
                new_fitness = self.fitness_function(new_solution)

                self.employed_bees[i].solution = new_solution
                self.employed_bees[i].fitness = new_fitness
                self.employed_bees[i].trial = 0

                if new_fitness > self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness

    def _generate_random_solution(self):
        solution = list(range(self.lb, self.ub + 1))
        random.shuffle(solution)
        return solution

    def plot_convergence(self, history):
        plt.plot(history)
        plt.title("График сходимости")
        plt.xlabel("Итерация")
        plt.ylabel("Разница с оптимумом")
        plt.show()
