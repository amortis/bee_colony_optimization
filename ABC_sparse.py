import random
import time
from datetime import timedelta
import numpy as np
import concurrent.futures
import multiprocessing as mp
import matplotlib.pyplot as plt

from bees import EmployedBee, OnlookerBee

# --- НОВАЯ ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ ФИТНЕСА В ОТДЕЛЬНОМ ПРОЦЕССЕ ---
def compute_fitness_task(args):
    solution, fitness_func = args
    return fitness_func(solution)


class ABCAlgorithmSparse:
    def __init__(self, fitness_function, distance_matrix, num_employed_bees, 
                 num_onlooker_bees, limit, patience, num_workers=None, seed=42):
        """
        Инициализация алгоритма для неполного графа.
        
        :param fitness_function: Функция, которая оценивает качество решения.
        :param distance_matrix: Матрица расстояний (могут быть значения np.inf).
        :param num_employed_bees: Количество рабочих пчел.
        :param num_onlooker_bees: Количество пчел-наблюдателей.
        :param limit: Максимальное количество неудачных попыток улучшения решения.
        """
        # Фиксируем случайность
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        
        # --- НОВОЕ: сохраняем матрицу расстояний ---
        self.distance_matrix = np.array(distance_matrix)
        self.num_cities = len(distance_matrix)
        
        # Проверяем матрицу на наличие np.inf
        self.has_infinite_distances = np.any(np.isinf(self.distance_matrix))
        
        # --- НОВОЕ: границы теперь от 0 до n-1 ---
        self.lb = 0
        self.ub = self.num_cities - 1
        
        self.num_workers = num_workers or min(32, mp.cpu_count())
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
        
        self.fitness_function = fitness_function
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
        self.global_history = []

        # Для вычисления итераций без улучшения
        self.patience = patience
        self.wait = 0
        self.best_iteration = 0

        # Отсчет времени
        self.start_time = None
        self.end_time = None

    def run_algorithm(self, max_iterations, initial_cycle=None):
        """
        Полный цикл выполнения алгоритма.
        
        :param initial_cycle: Гарантированный гамильтонов цикл для инициализации
        """
        try:
            self._initialize_population(initial_cycle)
            self.start_time = time.time()

            for iteration in range(max_iterations):
                old_best = self.best_fitness
                self.employed_bee_phase()
                self.onlooker_bee_phase()
                self.scout_bee_phase()
                
                # Сохраняем длину лучшего маршрута для истории
                if self.best_fitness > 0:
                    self.global_history.append(1 / self.best_fitness)

                if self.best_fitness > old_best:
                    self.wait = 0
                    self.best_iteration = iteration
                else:
                    self.wait += 1

                if self.wait >= self.patience:
                    print(f"\nEarly stopping at iteration {iteration}")
                    print(f"No improvement for {self.patience} iterations")
                    break

                if iteration % 20 == 0 and self.best_fitness > 0:
                    elapsed = self.get_formatted_time()
                    print(f"Iteration {iteration}. Time: {elapsed}. Best distance = {1 / self.best_fitness:.2f}")

            self.plot_convergence()
            return self.best_solution, self.best_fitness
        finally:
            self.executor.shutdown(wait=True)

    def get_formatted_time(self, seconds=None):
        """Форматирует время в читаемый вид (HH:MM:SS)"""
        if seconds is None:
            seconds = self.get_elapsed_time()
        return str(timedelta(seconds=seconds)).split(".")[0]

    def get_elapsed_time(self):
        """Возвращает время выполнения в секундах"""
        return time.time() - self.start_time

    # --- ИЗМЕНЕННАЯ ИНИЦИАЛИЗАЦИЯ С УЧЕТОМ ГАРАНТИРОВАННОГО ЦИКЛА ---
    def _initialize_population(self, initial_cycle=None):
        """
        Инициализация начальной популяции пчел.
        Если передан initial_cycle, используем его для части пчел.
        """
        initial_solutions = []
        
        # Если есть гарантированный цикл, используем его
        if initial_cycle is not None:
            # Проверяем, что цикл допустим
            if self._is_valid_cycle(initial_cycle):
                initial_solutions.append(initial_cycle)
                print(f"Используем гарантированный цикл длины: {1/self.fitness_function(initial_cycle):.2f}")
                
                # Дублируем цикл для нескольких пчел
                for _ in range(min(3, self.num_employed_bees - 1)):
                    initial_solutions.append(initial_cycle.copy())
        
        # Генерируем случайные допустимые решения для остальных пчел
        needed_solutions = self.num_employed_bees - len(initial_solutions)
        for _ in range(needed_solutions):
            solution = self._generate_valid_solution()
            initial_solutions.append(solution)

        # Подготовка задач для пула
        tasks = [(sol, self.fitness_function) for sol in initial_solutions]

        # Вычисление фитнеса в пуле
        fitness_results = list(self.executor.map(compute_fitness_task, tasks))

        # Создание пчёл
        for i, solution in enumerate(initial_solutions):
            employed_bee = EmployedBee(solution, self.fitness_function, 
                                       initial_fitness=fitness_results[i])
            self.employed_bees.append(employed_bee)

            # Обновление лучшего решения
            if employed_bee.fitness > self.best_fitness:
                self.best_solution = employed_bee.solution
                self.best_fitness = employed_bee.fitness

    # --- НОВЫЙ МЕТОД: ГЕНЕРАЦИЯ ДОПУСТИМОГО РЕШЕНИЯ ---
    def _generate_valid_solution(self, max_attempts=1000):
        """
        Генерирует допустимое решение (гамильтонов цикл) для неполного графа.
        Использует эвристику ближайшего соседа.
        """
        # Начинаем со случайного города
        start_city = random.randint(0, self.num_cities - 1)
        solution = [start_city]
        visited = set([start_city])
        
        # Строим цикл жадным алгоритмом
        current_city = start_city
        for _ in range(self.num_cities - 1):
            # Находим всех непосещенных соседей
            neighbors = []
            for city in range(self.num_cities):
                if (city not in visited and 
                    not np.isinf(self.distance_matrix[current_city][city]) and
                    city != current_city):
                    neighbors.append(city)
            
            if not neighbors:
                # Если нет доступных соседей, начинаем заново
                return self._generate_random_solution()
            
            # Выбираем случайного соседа (можно использовать ближайшего)
            next_city = random.choice(neighbors)
            solution.append(next_city)
            visited.add(next_city)
            current_city = next_city
        
        # Проверяем, можно ли вернуться в начальный город
        if np.isinf(self.distance_matrix[solution[-1]][solution[0]]):
            # Пытаемся найти другой путь или перестраиваем
            return self._generate_valid_solution(max_attempts - 1)
        
        return solution

    # --- НОВЫЙ МЕТОД: ПРОВЕРКА ЦИКЛА ---
    def _is_valid_cycle(self, solution):
        """Проверяет, является ли цикл допустимым для неполного графа"""
        if len(set(solution)) != self.num_cities:
            return False
            
        for i in range(len(solution)):
            city_from = solution[i]
            city_to = solution[(i + 1) % len(solution)]
            if np.isinf(self.distance_matrix[city_from][city_to]):
                return False
        return True

    # --- ОСТАВШИЕСЯ МЕТОДЫ ОСТАЮТСЯ ПРЕЖНИМИ, НО С КОРРЕКТИРОВКОЙ ---
    def employed_bee_phase(self):
        """Фаза занятых пчел"""
        futures = []
        for bee in self.employed_bees:
            future = self.executor.submit(bee.explore_async, 
                                         self.employed_bees, 
                                         self.fitness_function)
            futures.append(future)

        results = [future.result() for future in futures]

        for i, (is_improved, new_solution, new_fitness) in enumerate(results):
            bee = self.employed_bees[i]
            if is_improved and self._is_valid_cycle(new_solution):
                bee.solution = new_solution
                bee.fitness = new_fitness
                bee.trial = 0
                if new_fitness > self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness
            else:
                bee.trial += 1

    def onlooker_bee_phase(self):
        """Фаза пчел-наблюдателей"""
        if not self.employed_bees:
            return

        self.onlooker_bees.clear()
        employed_solutions = [bee.solution for bee in self.employed_bees]

        # Создаём задачи для onlooker bees
        onlooker_tasks = []
        for _ in range(self.num_onlooker_bees):
            initial_solution = random.choice(employed_solutions)
            onlooker_tasks.append((initial_solution, employed_solutions, 
                                  self.fitness_function))

        # Запускаем задачи асинхронно
        futures = [self.executor.submit(OnlookerBee.explore_async_static, task) 
                  for task in onlooker_tasks]

        # Сбор результатов
        results = [future.result() for future in futures]

        # Создание объектов OnlookerBee с проверкой допустимости
        self.onlooker_bees = []
        for solution, fitness in results:
            if self._is_valid_cycle(solution):
                onlooker_bee = OnlookerBee(solution, self.fitness_function, 
                                           initial_fitness=fitness)
                self.onlooker_bees.append(onlooker_bee)

        self._upgrade_solutions()

    def _upgrade_solutions(self):
        """Обновляет лучшее решение после этапа пчел наблюдателей"""
        if not self.onlooker_bees:
            return
            
        best_onlooker = max(self.onlooker_bees, key=lambda b: b.fitness)
        worst_employed_idx = min(range(len(self.employed_bees)), 
                                key=lambda i: self.employed_bees[i].fitness)
        
        if best_onlooker.fitness > self.employed_bees[worst_employed_idx].fitness:
            self.employed_bees[worst_employed_idx].solution = best_onlooker.solution.copy()
            self.employed_bees[worst_employed_idx].fitness = best_onlooker.fitness
            self.employed_bees[worst_employed_idx].trial = 0
        
        # Обновляем глобальное лучшее решение
        for bee in self.onlooker_bees:
            if bee.fitness > self.best_fitness:
                self.best_solution = bee.solution.copy()
                self.best_fitness = bee.fitness

    def scout_bee_phase(self):
        """Фаза разведчиков"""
        scouts_to_update = []
        for i, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                scouts_to_update.append(i)

        if not scouts_to_update:
            return

        # Генерируем новые допустимые решения
        new_solutions = [self._generate_valid_solution() for _ in scouts_to_update]

        # Подготовка задач для вычисления фитнеса
        tasks = [(sol, self.fitness_function) for sol in new_solutions]

        # Вычисление фитнеса в пуле
        new_fitnesses = list(self.executor.map(compute_fitness_task, tasks))

        # Обновление пчёл
        for idx, sol, fit in zip(scouts_to_update, new_solutions, new_fitnesses):
            if self._is_valid_cycle(sol):
                bee = self.employed_bees[idx]
                bee.solution = sol
                bee.fitness = fit
                bee.trial = 0

                if fit > self.best_fitness:
                    self.best_solution = sol.copy()
                    self.best_fitness = fit

    def _generate_random_solution(self):
        """Генерирует случайную перестановку (может быть недопустимой)"""
        solution = list(range(self.num_cities))
        random.shuffle(solution)
        return solution

    def plot_convergence(self):
        """Визуализация сходимости"""
        if not self.global_history:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.global_history)
        plt.title("График сходимости (длина лучшего маршрута)")
        plt.xlabel("Итерация")
        plt.ylabel("Длина маршрута")
        plt.grid(True)
        plt.show()