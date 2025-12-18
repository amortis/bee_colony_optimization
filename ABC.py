import random
import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing as mp

# Импортируем оптимизированные функции для TSP
try:
    from tsp_optimizations import nearest_neighbor_init, greedy_init, local_search_2opt, two_opt_swap, calculate_distance
    from bees.employedBee import compute_fitness_from_distance_matrix
    TSP_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    TSP_OPTIMIZATIONS_AVAILABLE = False
    def compute_fitness_from_distance_matrix(solution, distance_matrix):
        if distance_matrix is None:
            return 0.0
        total_distance = 0
        n = len(solution)
        for i in range(n):
            city1 = solution[i]
            city2 = solution[(i + 1) % n]
            total_distance += distance_matrix[city1, city2]
        return 1.0 / total_distance if total_distance > 0 else 0.0

from bees import EmployedBee, OnlookerBee

# --- ТОП-УРОВНЕВАЯ ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ ФИТНЕСА В ОТДЕЛЬНОМ ПРОЦЕССЕ ---
def compute_fitness_task(args):
    solution, distance_matrix = args
    return compute_fitness_from_distance_matrix(solution, distance_matrix)

# --- ТОП-УРОВНЕВАЯ ФУНКЦИЯ ДЛЯ СИНХРОННОГО ФИТНЕСА ---
# Используем класс-обертку для передачи distance_matrix
class FitnessWrapper:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
    
    def __call__(self, tour):
        return compute_fitness_from_distance_matrix(tour, self.distance_matrix)

class ABCAlgorithm:
    def __init__(self, fitness_function=None, lb=0, ub=0, num_employed_bees=50, num_onlooker_bees=50, limit=100, patience=100, optimal_length=0, distance_matrix=None, visualization=False, num_workers=None, seed=42,
                 local_search_interval=20, local_search_iterations=200, elitism_rate=0.1, heuristic_init_ratio=0.7):
        """
        Инициализация алгоритма.

        :param fitness_function: Функция, которая оценивает качество решения (опционально, если есть distance_matrix).
        :param lb: Нижняя граница пространства решений.
        :param ub: Верхняя граница пространства решений.
        :param num_employed_bees: Количество рабочих пчел.
        :param num_onlooker_bees: Количество пчел-наблюдателей.
        :param limit: Максимальное количество неудачных попыток улучшения решения.
        :param distance_matrix: Матрица расстояний для TSP (если указана, используется вместо fitness_function).
        """
        # проверки
        assert ub > lb, "Верхняя граница должна быть больше нижней"
        assert num_employed_bees > 0, "Должна быть хотя бы одна рабочая пчела"
        assert num_onlooker_bees > 0, "Должна быть хотя бы одна пчела-наблюдатель"
        assert limit > 0, "Лимит неудач должен быть положительным"

        # --- НОВОЕ ---
        self.num_workers = num_workers or min(32, mp.cpu_count()) # По умолчанию используем количество ядер
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
        # --- КОНЕЦ НОВОГО ---

        # Преобразуем в numpy array для эффективности (из ILS)
        self.distance_matrix = np.asarray(distance_matrix, dtype=np.float64) if distance_matrix is not None else None
        
        # ИЗМЕНЕНО: Используем класс-обертку вместо lambda
        if self.distance_matrix is not None:
            self.fitness_function = FitnessWrapper(self.distance_matrix)
        elif fitness_function is not None:
            self.fitness_function = fitness_function
        else:
            raise ValueError("Необходимо указать либо distance_matrix, либо fitness_function")
        
        self.lb = lb
        self.ub = ub
        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.limit = limit
        self.optimal_length = optimal_length
        self.visualization = visualization
        # Параметры локального поиска и элитизма
        self.local_search_interval = local_search_interval
        self.local_search_iterations = local_search_iterations
        self.elitism_rate = elitism_rate
        self.heuristic_init_ratio = heuristic_init_ratio  # Доля решений, инициализированных эвристиками
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

    # --- ОПТИМИЗИРОВАННОЕ ВЫЧИСЛЕНИЕ РАССТОЯНИЯ (из ILS) ---
    def _calculate_distance(self, tour):
        """
        Вычисляет общую длину маршрута (оптимизированная версия из ILS).
        Использует numpy для быстрого доступа к матрице расстояний.
        """
        if self.distance_matrix is None:
            return 0
        
        distance = 0
        n = len(tour)
        for i in range(n):
            city1 = tour[i]
            city2 = tour[(i + 1) % n]
            distance += self.distance_matrix[city1, city2]
        return distance

    # --- ОПЕРАТОР ВОЗМУЩЕНИЯ (из ILS) ---
    def _perturbation(self, tour, strength=3):
        """
        Оператор возмущения (перемешивания) для ILS.
        Выполняет 'strength' случайных 2-opt обменов.
        Полезно для разведчиков и выхода из локальных оптимумов.
        """
        new_tour = tour[:]
        n = len(new_tour)
        for _ in range(strength):
            # Выбираем два случайных индекса
            i, k = sorted(random.sample(range(n), 2))
            if i == k: 
                continue
            # Выполняем 2-opt обмен (инверсию подмаршрута)
            new_tour[i:k+1] = new_tour[i:k+1][::-1]
        return new_tour

    def _two_opt_swap(self, tour, i, k):
        """Выполняет 2-opt обмен (инверсию подмаршрута) - из ILS."""
        new_tour = tour[:]
        new_tour[i:k+1] = new_tour[i:k+1][::-1]
        return new_tour

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

                # Элитизм: периодически копируем лучшее решение в часть популяции
                if self.elitism_rate > 0 and iteration % self.local_search_interval == 0:
                    self._apply_elitism()

                # Локальный поиск вокруг текущего лучшего решения
                if self.local_search_interval > 0 and iteration % self.local_search_interval == 0:
                    self._local_2opt_search_global()

                # Проверка на None для distance_matrix
                if self.distance_matrix is not None:
                    current_distance = 1.0 / self.best_fitness if self.best_fitness > 0 else float('inf')
                    self.global_history.append(current_distance - self.optimal_length)

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
                    best_dist_str = f"{1 / self.best_fitness:.2f}" if self.best_fitness > 0 else "inf"
                    print(f"Iteration {iteration}. Time: {elapsed}. Best distance = {best_dist_str}")
            if self.visualization:
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
        Использует смешанную стратегию: часть решений генерируется эвристиками, часть случайно.
        """
        # Генерация начальных решений
        initial_solutions = []
        
        num_heuristic = int(self.num_employed_bees * self.heuristic_init_ratio)
        num_random = self.num_employed_bees - num_heuristic
        
        # Эвристическая инициализация (если есть матрица расстояний)
        # ВАЖНО: Добавляем случайные обмены к эвристическим решениям, чтобы они не были слишком хорошими
        if self.distance_matrix is not None and TSP_OPTIMIZATIONS_AVAILABLE:
            for i in range(num_heuristic):
                if i % 3 == 0:
                    solution = nearest_neighbor_init(self.distance_matrix, random_start=False)
                elif i % 3 == 1:
                    solution = greedy_init(self.distance_matrix)
                else:
                    solution = nearest_neighbor_init(self.distance_matrix, random_start=True)
                
                # "Портим" эвристическое решение случайными обменами, чтобы оно не было слишком хорошим
                # Применяем несколько случайных 2-opt обменов
                num_swaps = max(1, len(solution) // 20)  # Примерно 5% от размера
                for _ in range(num_swaps):
                    idx1, idx2 = sorted(random.sample(range(len(solution)), 2))
                    solution[idx1:idx2+1] = solution[idx1:idx2+1][::-1]
                
                initial_solutions.append(solution)
        elif self.distance_matrix is not None:
            # Fallback на старые методы если модуль не доступен
            for i in range(num_heuristic):
                if i % 3 == 0:
                    solution = self._nearest_neighbor_init()
                elif i % 3 == 1:
                    solution = self._greedy_init()
                else:
                    solution = self._nearest_neighbor_init(random_start=True)
                
                # "Портим" эвристическое решение случайными обменами
                num_swaps = max(1, len(solution) // 20)
                for _ in range(num_swaps):
                    idx1, idx2 = sorted(random.sample(range(len(solution)), 2))
                    solution[idx1:idx2+1] = solution[idx1:idx2+1][::-1]
                
                initial_solutions.append(solution)
        else:
            # Если матрицы нет, все случайные
            num_random = self.num_employed_bees
        
        # Случайная инициализация
        for _ in range(num_random):
            solution = random.sample(range(self.lb, self.ub + 1), self.ub - self.lb + 1)
            initial_solutions.append(solution)

        # Подготовка задач для пула
        tasks = [(sol, self.distance_matrix) for sol in initial_solutions]

        # Вычисление фитнеса в пуле
        fitness_results = list(self.executor.map(compute_fitness_task, tasks))

        # Создание пчёл
        for i, solution in enumerate(initial_solutions):
            employed_bee = EmployedBee(solution, self.fitness_function, initial_fitness=fitness_results[i], distance_matrix=self.distance_matrix)
            self.employed_bees.append(employed_bee)

            # Обновление лучшего решения
            if employed_bee.fitness > self.best_fitness:
                self.best_solution = employed_bee.solution
                self.best_fitness = employed_bee.fitness
        
        # Выводим начальное лучшее решение для контроля
        if self.distance_matrix is not None:
            initial_distance = 1.0 / self.best_fitness if self.best_fitness > 0 else float('inf')
            print(f"Начальное лучшее расстояние: {initial_distance:.2f} (оптимум: {self.optimal_length})")
            if self.optimal_length > 0:
                initial_gap = ((initial_distance - self.optimal_length) / self.optimal_length) * 100
                print(f"Начальный gap: {initial_gap:.2f}%")

    # --- ИЗМЕНЁННАЯ ФАЗА РАБОЧИХ ПЧЁЛ ---
    def employed_bee_phase(self) -> None:
        """
        Фаза занятых пчел: рабочие пчелы улучшают свои решения с многопоточностью.
        """
        futures = []
        for bee in self.employed_bees:
            # Подготовка задачи для explore
            # explore теперь возвращает (is_improved, new_solution, new_fitness)
            future = self.executor.submit(bee.explore_async, self.employed_bees, self.distance_matrix)
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
            initial_solution = random.choice(employed_solutions)
            onlooker_tasks.append((initial_solution, employed_solutions, self.distance_matrix))

        # Запускаем задачи асинхронно
        futures = [self.executor.submit(OnlookerBee.explore_async_static_wrapper, task) for task in onlooker_tasks]

        # Сбор результатов
        results = [future.result() for future in futures]

        # Создание объектов OnlookerBee с обновлёнными данными
        for solution, fitness in results:
            onlooker_bee = OnlookerBee(solution, self.fitness_function, initial_fitness=fitness, distance_matrix=self.distance_matrix)
            self.onlooker_bees.append(onlooker_bee)

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
        Использует оператор возмущения из ILS для более интеллектуального поиска.
        """
        scouts_to_update = []
        for i, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                scouts_to_update.append(i)

        if not scouts_to_update:
             return # Нечего обновлять

        # Генерируем новые решения (смешанная стратегия: часть случайных, часть с возмущением)
        new_solutions = []
        for idx in scouts_to_update:
            bee = self.employed_bees[idx]
            # 50% - случайное решение, 50% - возмущение лучшего решения
            if random.random() < 0.5 and self.best_solution is not None:
                # Используем оператор возмущения из ILS для выхода из локальных оптимумов
                perturbation_strength = max(3, len(self.best_solution) // 10)
                new_solution = self._perturbation(self.best_solution, strength=perturbation_strength)
            else:
                new_solution = self._generate_random_solution()
            new_solutions.append(new_solution)

        # Подготовка задач для вычисления фитнеса
        tasks = [(sol, self.distance_matrix) for sol in new_solutions]

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

    def _generate_random_solution(self):
        """Генерирует полностью случайное решение для задачи коммивояжера"""
        solution = list(range(self.lb, self.ub + 1))
        random.shuffle(solution)
        return solution

    # --- ЭВРИСТИКИ ИНИЦИАЛИЗАЦИИ ---
    def _nearest_neighbor_init(self, random_start=False):
        """
        Nearest Neighbor эвристика (улучшенная версия из ILS): 
        начинаем с случайного города и всегда идём к ближайшему непосещённому.
        
        :param random_start: Если True, стартовый город выбирается случайно, иначе с 0.
        :return: Маршрут, построенный по эвристике ближайшего соседа.
        """
        if self.distance_matrix is None:
            return self._generate_random_solution()
        
        n = self.ub - self.lb + 1
        unvisited = set(range(self.lb, self.ub + 1))
        tour = []
        
        # Выбираем стартовый город (как в ILS)
        if random_start:
            start_node = random.randint(self.lb, self.ub)
        else:
            start_node = self.lb
        
        tour.append(start_node)
        unvisited.remove(start_node)
        current_node = start_node
        
        # Строим тур, всегда выбирая ближайший непосещённый город (оптимизированный доступ через numpy)
        while unvisited:
            min_dist = float('inf')
            next_node = -1
            for neighbor in unvisited:
                dist = self.distance_matrix[current_node, neighbor]
                if dist < min_dist:
                    min_dist = dist
                    next_node = neighbor
            tour.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
        
        return tour

    def _greedy_init(self):
        """
        Упрощённая Greedy эвристика: строим тур, на каждом шаге выбирая ближайший город,
        но с учётом уже построенной части тура (выбираем из ближайших к любому концу).
        """
        if self.distance_matrix is None:
            return self._generate_random_solution()
        
        n = self.ub - self.lb + 1
        unvisited = set(range(self.lb, self.ub + 1))
        tour = []
        
        # Начинаем со случайного города
        start = random.choice(list(unvisited))
        tour.append(start)
        unvisited.remove(start)
        
        # Строим тур, на каждом шаге выбирая ближайший к любому концу тура город
        while unvisited:
            if len(tour) == 1:
                # Если только один город, выбираем ближайший к нему
                nearest = min(unvisited, key=lambda city: self.distance_matrix[tour[0], city])
                tour.append(nearest)
                unvisited.remove(nearest)
            else:
                # Выбираем ближайший к любому концу тура (началу или концу)
                start_city = tour[0]
                end_city = tour[-1]
                
                nearest_to_start = min(unvisited, key=lambda city: self.distance_matrix[start_city, city])
                nearest_to_end = min(unvisited, key=lambda city: self.distance_matrix[end_city, city])
                
                dist_to_start = self.distance_matrix[start_city, nearest_to_start]
                dist_to_end = self.distance_matrix[end_city, nearest_to_end]
                
                if dist_to_start < dist_to_end:
                    tour.insert(0, nearest_to_start)
                    unvisited.remove(nearest_to_start)
                else:
                    tour.append(nearest_to_end)
                    unvisited.remove(nearest_to_end)
        
        return tour

    # --- ЛОКАЛЬНЫЙ ПОИСК И ЭЛИТИЗМ ---
    def _two_opt_move(self, solution):
        """Один шаг 2-opt над маршрутом."""
        size = len(solution)
        i, j = sorted(random.sample(range(size), 2))
        new_solution = solution.copy()
        new_solution[i:j + 1] = reversed(new_solution[i:j + 1])
        return new_solution

    # --- ЛОКАЛЬНЫЙ ПОИСК И ЭЛИТИЗМ (ИСПОЛЬЗУЕМ НОВЫЙ 2-OPT) ---
    def _local_2opt_search_global(self):
        """
        Применяет высокоэффективный 2-opt локальный поиск к лучшему решению.
        """
        if self.best_solution is None or self.distance_matrix is None:
            return

        if TSP_OPTIMIZATIONS_AVAILABLE:
            # Используем импортированную функцию local_search_2opt
            new_solution, new_distance = local_search_2opt(self.best_solution, self.distance_matrix)
            
            # Фитнес - это обратная величина расстояния
            new_fitness = 1.0 / new_distance
        else:
            # Fallback на старый метод
            current_solution = self.best_solution.copy()
            n = len(current_solution)
            
            while True:
                best_improvement = 0
                best_i, best_k = -1, -1

                for i in range(n - 1):
                    for k in range(i + 1, n):
                        i_prev = (i - 1) % n
                        k_next = (k + 1) % n

                        A = current_solution[i_prev]
                        B = current_solution[i]
                        C = current_solution[k]
                        D = current_solution[k_next]

                        if i_prev == k or i == k_next:
                            continue

                        old_dist = self.distance_matrix[A, B] + self.distance_matrix[C, D]
                        new_dist = self.distance_matrix[A, C] + self.distance_matrix[B, D]
                        
                        improvement = old_dist - new_dist

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_i, best_k = i, k

                if best_improvement > 0:
                    current_solution[best_i:best_k+1] = list(reversed(current_solution[best_i:best_k+1]))
                else:
                    break

            new_solution = current_solution
            new_fitness = self.fitness_function(new_solution)

        # Если локальный поиск улучшил глобальное лучшее — обновляем и часть популяции
        if new_fitness > self.best_fitness:
            self.best_solution = new_solution
            self.best_fitness = new_fitness

            # Немного распространяем улучшение на популяцию
            num_to_update = max(1, int(self.elitism_rate * len(self.employed_bees)))
            
            # Находим худших пчел для замены
            worst_bees = sorted(self.employed_bees, key=lambda b: b.fitness)[:num_to_update]
            
            for bee in worst_bees:
                bee.solution = new_solution.copy()
                bee.fitness = new_fitness
                bee.trial = 0

    def _apply_elitism(self):
        """
        Простая стратегия элитизма: часть худших рабочих пчёл заменяется копией лучшего решения.
        """
        if self.best_solution is None or self.elitism_rate <= 0:
            return

        num_to_replace = max(1, int(self.elitism_rate * len(self.employed_bees)))
        worst_bees = sorted(self.employed_bees, key=lambda b: b.fitness)[:num_to_replace]
        for bee in worst_bees:
            bee.solution = self.best_solution.copy()
            bee.fitness = self.best_fitness
            bee.trial = 0


    def plot_convergence(self, history):
        """
        Визуализация
        """
        plt.plot(history)
        plt.title("График сходимости")
        plt.xlabel("Итерация")
        plt.ylabel("Разница с оптимумом")
        plt.show()
