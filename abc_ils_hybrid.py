import random
import time
from datetime import timedelta
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

from bees import EmployedBee, OnlookerBee
from tsp_optimizations import (
    calculate_tour_length,
    local_search_2opt,
    local_search_3opt,
    fast_2opt,
    fast_2opt_neighbors,
    get_nearest_neighbors,
    nearest_neighbor_init,
    greedy_init,
    double_bridge_perturbation,
    perturbation_2opt_random,
    build_candidate_lists,
    lin_kernighan_simplified,
    multi_insert_perturbation,
    random_subsequence_reverse,
    random_subsequence_swap,
)


Tour = List[int]


class ABCTSPILS:
    """
    Гибридный алгоритм: ABC (Artificial Bee Colony) + идеи из ILS для TSP.

    - Глобальный поиск: популяционный ABC (employed / onlooker / scout).
    - Локальный поиск: эффективный 2-opt над лучшим туром через интервалы.
    - Инициализация: смесь nearest-neighbor и greedy эвристик.
    - Возмущение: double-bridge / случайные 2-opt при больших trial.
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        num_employed_bees: int = 50,
        num_onlooker_bees: int = 50,
        limit: int = 100,
        patience: int = 200,
        local_search_interval: int = 50,
        heuristic_init_ratio: float = 0.7,
        use_parallel: bool = True,
        num_workers: int = None,
        use_gpu: bool = False,
        optimal_value: Optional[float] = None,
    ):
        # матрица расстояний как numpy float64 (из ILS)
        self.distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        self.num_cities = self.distance_matrix.shape[0]
        self.optimal_value = optimal_value  # Оптимальное значение для визуализации
        
        # GPU поддержка
        self.use_gpu = use_gpu
        self.distance_matrix_gpu = None
        self.neighbors_gpu = None
        
        if self.use_gpu:
            try:
                from tsp_optimizations import GPU_AVAILABLE, cp
                if GPU_AVAILABLE:
                    # Копируем матрицу на GPU
                    self.distance_matrix_gpu = cp.asarray(self.distance_matrix, dtype=cp.float64)
                    print(f" GPU активирован (CuPy). Матрица расстояний загружена на GPU.")
                    print(f"   GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
                else:
                    print("  CuPy не доступна. Используется CPU.")
                    self.use_gpu = False
            except Exception as e:
                print(f"️  Ошибка при инициализации GPU: {e}. Используется CPU.")
                self.use_gpu = False

        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.limit = limit
        self.patience = patience
        self.local_search_interval = max(1, local_search_interval)
        self.heuristic_init_ratio = min(1.0, max(0.0, heuristic_init_ratio))
        
        # Параллелизация
        self.use_parallel = use_parallel
        self.num_workers = num_workers

        self.employed_bees: List[EmployedBee] = []
        self.onlooker_bees: List[OnlookerBee] = []

        self.best_tour: Tour | None = None
        self.best_distance: float = float("inf")

        self.history: List[float] = []

        self.start_time: float | None = None
        self.end_time: float | None = None
        self.wait = 0
        self.best_iteration = 0
        
        # Adaptive Diversity Control
        self.historical_best_solutions: List[Tour] = []
        self.max_historical_solutions = 5
        
        # Гибридный локальный поиск
        self._ls_improvement_history: List[float] = []
        
        # Memory-based подход (edge memory)
        self.edge_memory: np.ndarray = np.zeros((self.num_cities, self.num_cities))
        self.memory_decay = 0.95
        
        # Списки ближайших соседей для ускорения 2-opt (критично для 300-400 городов)
        # Сложность падает с O(N²) до O(N × n_neighbors)
        self.neighbors: Optional[np.ndarray] = None
        # Адаптивное количество соседей: больше для больших задач
        if self.num_cities >= 500:
            n_neighbors = 50  # Для очень больших задач (500+)
        elif self.num_cities >= 300:
            n_neighbors = 40  # Для больших задач (300-499)
        elif self.num_cities >= 200:
            n_neighbors = 30  # Для средних задач (200-299)
        else:
            n_neighbors = 20  # Для малых задач (<200)
        if self.num_cities > 50:  # Строим только для больших задач
            if self.use_gpu and self.distance_matrix_gpu is not None:
                from tsp_optimizations import get_nearest_neighbors_gpu, cp
                self.neighbors_gpu = get_nearest_neighbors_gpu(self.distance_matrix_gpu, n_neighbors=n_neighbors)
                # Также создаем CPU версию для совместимости
                self.neighbors = cp.asnumpy(self.neighbors_gpu)
                print(f"Построены списки ближайших соседей на GPU (n_neighbors={n_neighbors})")
            else:
                self.neighbors = get_nearest_neighbors(self.distance_matrix, n_neighbors=n_neighbors)
                print(f"Построены списки ближайших соседей (n_neighbors={n_neighbors}) для ускорения 2-opt")

    # ===================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ =====================

    def _fitness(self, tour: Tour) -> float:
        """
        Фитнес: чем меньше длина маршрута, тем лучше.
        Используем 1 / distance.
        """
        d = calculate_tour_length(self.distance_matrix, tour)
        # Чтобы избежать деления на ноль
        return 1.0 / (1e-9 + d)

    def get_formatted_time(self, seconds: float | None = None) -> str:
        if seconds is None:
            seconds = self.get_elapsed_time()
        return str(timedelta(seconds=int(seconds))).split(".")[0]

    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    # ===================== ИНИЦИАЛИЗАЦИЯ ПОПУЛЯЦИИ =====================

    def _initialize_population(self) -> None:
        """
        Умная инициализация популяции с использованием:
        - Истории лучших решений
        - Edge memory для guided nearest neighbor
        - Смеси эвристик и случайных решений
        """
        total = self.num_employed_bees
        num_heuristic = int(total * self.heuristic_init_ratio)

        solutions: List[Tour] = []

        # 1) Лучшие решения из истории (если есть)
        if self.historical_best_solutions:
            for sol in self.historical_best_solutions[:min(3, len(self.historical_best_solutions))]:
                solutions.append(sol.copy())

        # 2) Nearest neighbor с фиксированным стартом 0
        if num_heuristic > 0:
            solutions.append(nearest_neighbor_init(self.distance_matrix, start_city=0))

        # 3) Memory-guided nearest neighbor (использует edge memory)
        num_memory_guided = max(0, num_heuristic // 3)
        for _ in range(num_memory_guided):
            start = random.randint(0, self.num_cities - 1)
            solutions.append(self._memory_guided_nn(start))

        # 4) Остальные NN с случайным стартом
        while len(solutions) < num_heuristic:
            solutions.append(nearest_neighbor_init(self.distance_matrix, start_city=None))

        # 5) Greedy-эвристика (по крайней мере один раз)
        solutions.append(greedy_init(self.distance_matrix))

        # 6) Biased random tours (с bias к хорошим ребрам из памяти)
        num_biased = max(0, (total - len(solutions)) // 2)
        for _ in range(num_biased):
            solutions.append(self._biased_random_tour())

        # 7) Остальные — случайные перестановки
        while len(solutions) < total:
            tour = list(range(self.num_cities))
            random.shuffle(tour)
            solutions.append(tour)

        self.employed_bees.clear()
        self.best_tour = None
        self.best_distance = float("inf")

        for sol in solutions:
            bee = EmployedBee(sol, self._fitness, distance_matrix=self.distance_matrix)
            self.employed_bees.append(bee)

            tour_len = calculate_tour_length(self.distance_matrix, bee.solution)
            if tour_len < self.best_distance:
                self.best_distance = tour_len
                self.best_tour = bee.solution.copy()
        
        # Инициализируем edge memory из начальной популяции
        if self.best_tour is not None:
            self._update_edge_memory(self.best_tour)

    # ===================== ОСНОВНОЙ ЦИКЛ ABC+ILS =====================

    def run(self, max_iterations: int) -> Tuple[Tour, float]:
        """
        Запуск гибридного алгоритма.
        Возвращает (лучший_тур, его_длина).
        """
        self._initialize_population()
        self.start_time = time.time()
        self.history = []
        self.wait = 0
        self.best_iteration = 0

        assert self.best_tour is not None

        # Отслеживание застревания для глобального kick
        stagnation_counter = 0
        last_best_distance = self.best_distance
        
        for it in range(max_iterations):
            old_best = self.best_distance
            
            # ОТЛАДКА: что происходит в фазах ABC
            if it >= 20:
                # print(f"      Выполняем фазы ABC...")
                best_before_phases = self.best_distance

            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            
            #if it >= 20:
                #if self.best_distance + 1e-9 < best_before_phases:
                    #pass
                    # print(f"      ✅ Улучшение в фазах ABC: {best_before_phases:.2f} -> {self.best_distance:.2f}")
                # else:
                    # print(f"      ❌ Улучшения в фазах ABC нет (best={self.best_distance:.2f})")

            # Периодический глубокий 2-opt над лучшим решением (элитизм + интенсификация)
            # Делаем это чаще для лучшего качества, особенно при застревании
            base_interval = max(5, self.local_search_interval // 3) if self.num_cities >= 200 else max(10, self.local_search_interval // 2)
            if it > 0 and it % base_interval == 0:
                self._local_2opt_search_global()
            
            # Глубокий 2-opt - чаще для больших задач и при застревании
            deep_interval = 25 if self.num_cities >= 200 else 50

            if it > 0 and it % deep_interval == 0:
                #self._deep_local_search_global()
                pass

            self.history.append(self.best_distance)
            
            # ОТЛАДКА: что происходит на каждой итерации
            if it >= 20:  # Начинаем отладку с итерации 20

                if self.employed_bees:
                    distances = [calculate_tour_length(self.distance_matrix, bee.solution) for bee in self.employed_bees]

            
            # ПРИНУДИТЕЛЬНОЕ ВОЗМУЩЕНИЕ при застревании (ПЕРЕД проверкой улучшения!)
            # При stagnation > 10 делаем легкое возмущение части популяции
            if stagnation_counter > 10 and stagnation_counter % 5 == 0:
                # print(f"   🔄 Принудительное возмущение при застревании (stagnation={stagnation_counter})")
                old_best_before_perturb = self.best_distance
                self._force_light_perturbation()
                #if it >= 20:
                    # print(f"      После возмущения: best_distance = {self.best_distance:.2f} (было {old_best_before_perturb:.2f})")
                    # if self.best_distance + 1e-9 < old_best_before_perturb:
                        # print(f"      ✅ НАЙДЕНО УЛУЧШЕНИЕ в возмущении!")
                    # else:
                        # print(f"      ❌ Улучшения не найдено в возмущении")

            # Проверка улучшения ПОСЛЕ всех фаз (включая возмущения)
            # old_best был сохранен в начале итерации, поэтому сравниваем с ним
            #if it >= 20:
                # print(f"      Проверка улучшения: {self.best_distance:.2f} < {old_best:.2f}? {self.best_distance + 1e-9 < old_best}")
            
            if self.best_distance + 1e-9 < old_best:
                # if it >= 20:
                    # print(f"      ✅ УЛУЧШЕНИЕ НАЙДЕНО! {old_best:.2f} -> {self.best_distance:.2f}")
                self.wait = 0
                self.best_iteration = it
                stagnation_counter = 0  # Сброс счетчика застревания
                last_best_distance = self.best_distance
                # Обновляем память о хороших ребрах
                if self.best_tour is not None:
                    self._update_edge_memory(self.best_tour)
            else:
                # if it >= 20:
                    # print(f"      ❌ Улучшения нет. stagnation_counter увеличивается: {stagnation_counter} -> {stagnation_counter + 1}")
                self.wait += 1
                stagnation_counter += 1

            # ГЛОБАЛЬНЫЙ KICK при застревании (адаптивный порог)
            # Для задач 200-400 городов: более агрессивный kick, но не слишком рано
            kick_threshold = 40 if self.num_cities >= 200 else 50
            if stagnation_counter > kick_threshold:
                # print(f"\n{'='*60}")
                # print(f"⚠️  ГЛОБАЛЬНЫЙ KICK на итерации {it}")
                # print(f"   Застревание: {stagnation_counter} итераций")
                # print(f"   Текущий лучший результат: {self.best_distance:.2f}")
                # print(f"   Размер популяции: {len(self.employed_bees)} пчел")
                
                # Анализ популяции перед kick
                # if self.employed_bees:
                    # distances = [calculate_tour_length(self.distance_matrix, bee.solution) for bee in self.employed_bees]
                    # avg_dist = np.mean(distances)
                    # min_dist = np.min(distances)
                    # max_dist = np.max(distances)
                    # std_dist = np.std(distances)
                    # print(f"   Популяция перед kick:")
                    # print(f"      Среднее: {avg_dist:.2f}, Мин: {min_dist:.2f}, Макс: {max_dist:.2f}, Стд: {std_dist:.2f}")
                    # print(f"      Разнообразие (std/mean): {std_dist/avg_dist*100:.2f}%")
                
                old_best = self.best_distance
                self._global_kick()
                
                # Результаты после kick
                # if self.employed_bees:
                    # distances_after = [calculate_tour_length(self.distance_matrix, bee.solution) for bee in self.employed_bees]
                    # avg_dist_after = np.mean(distances_after)
                    # min_dist_after = np.min(distances_after)
                    # print(f"   Популяция после kick:")
                    # print(f"      Среднее: {avg_dist_after:.2f}, Мин: {min_dist_after:.2f}")
                    # print(f"      Изменение лучшего: {self.best_distance - old_best:.2f} ({'+' if self.best_distance > old_best else ''}{((self.best_distance - old_best) / old_best * 100):.2f}%)")
                
                # print(f"{'='*60}\n")
                stagnation_counter = 0
                last_best_distance = self.best_distance

            # Адаптация параметров
            if it > 0:
                self._adapt_parameters(it, max_iterations)
            
            # Adaptive Diversity Control (умеренная частота)
            diversity_check_interval = 15 if self.num_cities >= 200 else 20
            if it > 0 and it % diversity_check_interval == 0:
                self._adaptive_diversity_control()
            
            # Дополнительная проверка при застревании (только при сильном застревании)
            if stagnation_counter > 20 and it % 10 == 0:
                # if stagnation_counter == 21:  # Выводим только при первом срабатывании
                    # print(f"    Ранняя проверка разнообразия (stagnation={stagnation_counter})")
                self._adaptive_diversity_control()
            
            if self.wait >= self.patience:
                print(f"\nEarly stopping at iteration {it}, "
                      f"no improvement for {self.patience} iterations.")
                break

            if it % 30 == 0:
                print(
                    f"Iteration {it}: best distance = {self.best_distance:.2f}, "
                    f"time = {self.get_formatted_time()}, "
                    f"stagnation = {stagnation_counter}"
                )

        self.end_time = time.time()
        assert self.best_tour is not None
        print(f"\nFinished. Best distance = {self.best_distance:.2f}, "
              f"time = {self.get_formatted_time()}")
        
        #Автоматическая визуализация сходимости
        if len(self.history) > 0:
             self.plot_convergence(optimal_value=self.optimal_value)
        
        return self.best_tour, self.best_distance

    # ===================== ФАЗЫ ABC =====================

    def _process_employed_bee(self, bee: EmployedBee) -> Tuple[EmployedBee, bool, float]:
        """
        Обрабатывает одну занятую пчелу. Используется для параллелизации.
        """
        improved = False
        tour_len = calculate_tour_length(self.distance_matrix, bee.solution)
        
        # Если пчела давно не улучшалась — потрясём её маршрут
        if bee.trial > self.limit // 3:
            perturbed = double_bridge_perturbation(bee.solution)
            perturbed = perturbation_2opt_random(self.distance_matrix, perturbed, strength=2)
            improved = bee.update_solution(perturbed)
            if improved:
                bee.trial = 0
                tour_len = calculate_tour_length(self.distance_matrix, bee.solution)

        improved_flag = bee.explore(self.employed_bees)
        tour_len = calculate_tour_length(self.distance_matrix, bee.solution)
        
        if improved_flag:
            bee.trial = 0
        else:
            bee.trial += 1
            
        return bee, improved_flag or improved, tour_len

    def employed_bee_phase(self) -> None:
        """
        Фаза занятых пчёл с поддержкой параллелизации.
        Дополнительно: при большом trial для пчелы используем сильное возмущение (double-bridge).
        """
        if self.use_parallel and len(self.employed_bees) > 10:
            # Параллельная обработка
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self._process_employed_bee, bee): idx 
                          for idx, bee in enumerate(self.employed_bees)}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    bee, improved_flag, tour_len = future.result()
                    # Обновляем пчелу в списке
                    self.employed_bees[idx] = bee
                    if tour_len < self.best_distance:
                        self.best_distance = tour_len
                        self.best_tour = bee.solution.copy()
        else:
            # Последовательная обработка с улучшенным локальным поиском
            for bee in self.employed_bees:
                # Если пчела давно не улучшалась — потрясём её маршрут
                if bee.trial > self.limit // 3:
                    perturbed = double_bridge_perturbation(bee.solution)
                    perturbed = perturbation_2opt_random(self.distance_matrix, perturbed, strength=2)
                    improved = bee.update_solution(perturbed)
                    if improved:
                        bee.trial = 0

                improved_flag = bee.explore(self.employed_bees)

                # ИНТЕНСИФИКАЦИЯ: Применяем быстрый локальный поиск ВСЕГДА (вероятность 1.0)
                # Для задач 100-400 городов запас скорости есть, делаем 2-opt всегда
                # Это значительно улучшает качество решений
                if improved_flag:
                    if self.use_gpu and self.distance_matrix_gpu is not None:
                        from tsp_optimizations import fast_2opt_gpu, calculate_tour_length_gpu, cp
                        tour_gpu = cp.asarray(bee.solution, dtype=cp.int32)
                        optimized_gpu = fast_2opt_gpu(self.distance_matrix_gpu, tour_gpu, 
                                                      neighbors_gpu=self.neighbors_gpu)
                        optimized_list = list(cp.asnumpy(optimized_gpu))
                        optimized_len = calculate_tour_length_gpu(self.distance_matrix_gpu, optimized_gpu)
                        current_len = calculate_tour_length_gpu(self.distance_matrix_gpu, tour_gpu)
                    else:
                        tour_array = np.asarray(bee.solution, dtype=np.int32)
                        if self.neighbors is not None:
                            optimized = fast_2opt_neighbors(self.distance_matrix, tour_array, self.neighbors)
                        else:
                            optimized = fast_2opt(self.distance_matrix, tour_array)
                        optimized_list = list(optimized)
                        optimized_len = calculate_tour_length(self.distance_matrix, optimized_list)
                        current_len = calculate_tour_length(self.distance_matrix, bee.solution)
                    
                    if optimized_len < current_len:
                        bee.solution = optimized_list
                        bee.fitness = self._fitness(optimized_list)
                        if hasattr(bee, 'invalidate_cache'):
                            bee.invalidate_cache()

                tour_len = calculate_tour_length(self.distance_matrix, bee.solution)
                if tour_len < self.best_distance:
                    self.best_distance = tour_len
                    self.best_tour = bee.solution.copy()

                if improved_flag:
                    bee.trial = 0
                else:
                    bee.trial += 1

    def onlooker_bee_phase(self) -> None:
        """
        Фаза пчёл-наблюдателей.
        """
        if not self.employed_bees:
            return

        self.onlooker_bees.clear()
        solutions = [bee.solution for bee in self.employed_bees]

        for _ in range(self.num_onlooker_bees):
            onlooker = OnlookerBee(random.choice(solutions), self._fitness)
            improved = onlooker.explore(solutions)
            self.onlooker_bees.append(onlooker)

            tour_len = calculate_tour_length(self.distance_matrix, onlooker.solution)
            if tour_len < self.best_distance:
                self.best_distance = tour_len
                self.best_tour = onlooker.solution.copy()

        # элитизм: лучшие наблюдатели могут вытеснить худших занятых пчёл
        self._elitist_update_from_onlookers()

    def _elitist_update_from_onlookers(self) -> None:
        if not self.onlooker_bees:
            return

        best_onlooker = min(
            self.onlooker_bees,
            key=lambda b: calculate_tour_length(self.distance_matrix, b.solution),
        )
        worst_employed_idx = max(
            range(len(self.employed_bees)),
            key=lambda i: calculate_tour_length(self.distance_matrix, self.employed_bees[i].solution),
        )

        best_onlooker_len = calculate_tour_length(self.distance_matrix, best_onlooker.solution)
        worst_employed_len = calculate_tour_length(
            self.distance_matrix, self.employed_bees[worst_employed_idx].solution
        )

        if best_onlooker_len + 1e-9 < worst_employed_len:
            self.employed_bees[worst_employed_idx].solution = best_onlooker.solution.copy()
            self.employed_bees[worst_employed_idx].fitness = self._fitness(best_onlooker.solution)
            self.employed_bees[worst_employed_idx].trial = 0

    def scout_bee_phase(self) -> None:
        """
        Умная фаза разведчиков: вместо random.shuffle используем Double Bridge Kick от лучшего решения.
        Это позволяет "выпрыгнуть" из локального минимума, но остаться в зоне хороших решений.
        Идея ILS: Local Opt -> Perturb -> Local Opt
        """
        if self.best_tour is None:
            return

        for i, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                # ВМЕСТО random.shuffle: берем лучшее глобальное решение и сильно его ломаем (Kick)
                # Для больших задач используем более сильное возмущение
                if self.num_cities >= 200:
                    # Двойное возмущение для больших задач
                    candidate = double_bridge_perturbation(self.best_tour)
                    candidate = multi_insert_perturbation(candidate, num_cities_to_move=2)
                else:
                    # Double Bridge (4 разрезами) - сохраняет 95% хорошего пути, меняет только чуть-чуть
                    candidate = double_bridge_perturbation(self.best_tour)
                
                # Сразу применяем быстрый 2-opt с соседями, чтобы вернуть его в локальный минимум
                # (Идея ILS: Local Opt -> Perturb -> Local Opt)
                candidate_array = np.asarray(candidate, dtype=np.int32)
                if self.neighbors is not None:
                    optimized = fast_2opt_neighbors(self.distance_matrix, candidate_array, self.neighbors)
                else:
                    optimized = fast_2opt(self.distance_matrix, candidate_array)
                new_tour = list(optimized)

                self.employed_bees[i].solution = new_tour
                self.employed_bees[i].fitness = self._fitness(new_tour)
                self.employed_bees[i].trial = 0
                if hasattr(self.employed_bees[i], 'invalidate_cache'):
                    self.employed_bees[i].invalidate_cache()

                tour_len = calculate_tour_length(self.distance_matrix, new_tour)
                if tour_len < self.best_distance:
                    self.best_distance = tour_len
                    self.best_tour = new_tour.copy()

    # ===================== ADAPTIVE DIVERSITY CONTROL =====================
    
    def _adapt_parameters(self, iteration: int, max_iterations: int) -> None:
        """
        Адаптирует параметры алгоритма в процессе работы.
        """
        progress = iteration / max_iterations if max_iterations > 0 else 0.0
        
        # Уменьшаем частоту локального поиска при приближении к концу
        base_interval = 50
        self.local_search_interval = max(10, int(base_interval * (1 - progress * 0.5)))
        
        # Адаптируем patience на основе прогресса
        if len(self.history) >= 10:
            recent_improvement = self.history[-10] - self.history[-1]
            if recent_improvement < self.best_distance * 0.01:  # Прогресс замедлился
                self.patience = min(500, self.patience + 10)
            else:
                self.patience = max(50, int(self.patience * 0.95))
        
        # Адаптируем limit на основе разнообразия
        if len(self.employed_bees) > 0:
            fitness_values = [self._fitness(bee.solution) for bee in self.employed_bees]
            fitness_variance = np.var(fitness_values)
            if fitness_variance > 0:
                # Увеличиваем limit при низком разнообразии
                variance_ratio = fitness_variance / (np.mean(fitness_values) + 1e-9)
                self.limit = max(50, min(200, int(100 + 100 * (1 - variance_ratio))))

    def _adaptive_diversity_control(self) -> None:
        """
        Динамически управляет разнообразием популяции.
        Предотвращает преждевременную сходимость.
        """
        if not self.employed_bees:
            return
        
        # Вычисляем средний фитнес и дисперсию
        fitness_values = [self._fitness(bee.solution) for bee in self.employed_bees]
        current_avg_fitness = np.mean(fitness_values)
        fitness_variance = np.var(fitness_values)
        
        # Если популяция сходится (низкая дисперсия) - умеренный порог
        # Не делаем слишком агрессивно, чтобы не ломать хорошие решения
        threshold = self.best_distance * 0.008 if self.num_cities >= 200 else self.best_distance * 0.01
        if fitness_variance < threshold or fitness_variance < 1e-6:
            self._inject_diversity()
        
        # Адаптивный limit
        if self.best_distance > 0:
            variance_ratio = fitness_variance / (self.best_distance + 1e-9)
            self.limit = max(50, int(200 * (1 - min(1.0, variance_ratio))))
    
    def _inject_diversity(self) -> None:
        """
        Инъекция разнообразия при сходимости популяции.
        """
        if self.best_tour is None:
            return
        
        # Умеренная инъекция разнообразия (не слишком агрессивно)
        num_to_perturb = max(1, len(self.employed_bees) // 4) if self.num_cities >= 200 else max(1, len(self.employed_bees) // 5)
        for i in range(num_to_perturb):
            idx = (self.best_iteration + i) % len(self.employed_bees)
            # Для больших задач используем более сильные возмущения
            if self.num_cities >= 200:
                if random.random() < 0.8:  # 80% вероятность сильного возмущения
                    # Двойное возмущение для больших задач
                    perturbed = double_bridge_perturbation(self.best_tour)
                    perturbed = multi_insert_perturbation(perturbed, num_cities_to_move=3)
                    self.employed_bees[idx].solution = perturbed
                else:
                    self.employed_bees[idx].solution = self._random_restart()
            else:
                if random.random() < 0.7:  # 70% вероятность сильного возмущения
                    self.employed_bees[idx].solution = double_bridge_perturbation(self.best_tour)
                else:
                    self.employed_bees[idx].solution = self._random_restart()
            
            self.employed_bees[idx].fitness = self._fitness(self.employed_bees[idx].solution)
            self.employed_bees[idx].trial = 0
            if hasattr(self.employed_bees[idx], 'invalidate_cache'):
                self.employed_bees[idx].invalidate_cache()
    
    def _force_light_perturbation(self) -> None:
        """
        Принудительное легкое возмущение части популяции при застревании.
        Вызывается независимо от состояния популяции.
        """
        if self.best_tour is None:
            # print(f"       best_tour is None, пропускаем возмущение")
            return
        
        old_best_before = self.best_distance
        # print(f"         Состояние перед возмущением:")
        # print(f"         Лучшее расстояние: {old_best_before:.2f}")
        # print(f"         Размер популяции: {len(self.employed_bees)} пчел")
        
        # Возмущаем 30-40% популяции легким способом
        num_to_perturb = max(2, len(self.employed_bees) // 3)
        indices = random.sample(range(len(self.employed_bees)), num_to_perturb)
        # print(f"      Возмущаем {num_to_perturb} пчел (индексы: {indices})")
        
        improved_any = False
        improved_count = 0
        for idx in indices:
            # Легкое возмущение - только Double Bridge
            perturbed = double_bridge_perturbation(self.best_tour)
            
            # Применяем быстрый 2-opt для оптимизации
            perturbed_array = np.asarray(perturbed, dtype=np.int32)
            if self.neighbors is not None:
                optimized = fast_2opt_neighbors(self.distance_matrix, perturbed_array, self.neighbors)
            else:
                optimized = fast_2opt(self.distance_matrix, perturbed_array)
            
            new_tour = list(optimized)
            new_len = calculate_tour_length(self.distance_matrix, new_tour)
            old_len = calculate_tour_length(self.distance_matrix, self.employed_bees[idx].solution)
            
            self.employed_bees[idx].solution = new_tour
            self.employed_bees[idx].fitness = self._fitness(new_tour)
            self.employed_bees[idx].trial = 0
            if hasattr(self.employed_bees[idx], 'invalidate_cache'):
                self.employed_bees[idx].invalidate_cache()
            
            # Проверяем улучшение
            if new_len < self.best_distance:
                improvement = self.best_distance - new_len
                # print(f"      Пчела {idx}: улучшение! {self.best_distance:.2f} -> {new_len:.2f} (Δ={improvement:.2f})")
                self.best_distance = new_len
                self.best_tour = new_tour.copy()
                improved_any = True
                improved_count += 1
            # elif new_len < old_len:
                # print(f"      Пчела {idx}: улучшение локально {old_len:.2f} -> {new_len:.2f} (но не глобально, best={self.best_distance:.2f})")
        
        # print(f"      Результаты возмущения:")
        # print(f"         Улучшено пчел: {improved_count}/{num_to_perturb}")
        # print(f"         Глобальное улучшение: {' ДА' if improved_any else ' НЕТ'}")
        # if improved_any:
            # print(f"         Новое лучшее расстояние: {self.best_distance:.2f} (было {old_best_before:.2f}, Δ={old_best_before - self.best_distance:.2f})")
        # else:
            # print(f"         Лучшее расстояние не изменилось: {self.best_distance:.2f}")
    
    def _random_restart(self) -> Tour:
        """Генерирует новое случайное решение"""
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour
    
    def _update_edge_memory(self, tour: Tour) -> None:
        """
        Обновляет память о хороших ребрах из лучшего тура.
        """
        if self.best_distance <= 0:
            return
        
        # Увеличиваем вес ребер из лучшего тура
        weight = 1.0 / (self.best_distance + 1e-9)
        for i in range(len(tour)):
            city1, city2 = tour[i], tour[(i + 1) % len(tour)]
            # Обновляем в обе стороны (симметричная матрица)
            self.edge_memory[city1, city2] = max(
                self.edge_memory[city1, city2] * self.memory_decay, 
                weight
            )
            self.edge_memory[city2, city1] = self.edge_memory[city1, city2]
    
    def _memory_guided_nn(self, start_city: int) -> Tour:
        """
        Nearest neighbor с использованием edge memory для выбора следующего города.
        """
        n = self.num_cities
        tour: Tour = [start_city]
        unvisited = set(range(n))
        unvisited.remove(start_city)
        current = start_city

        while unvisited:
            # Комбинируем расстояние и память
            scores = []
            for j in unvisited:
                distance_score = 1.0 / (self.distance_matrix[current, j] + 1e-9)
                memory_score = self.edge_memory[current, j]
                # Взвешенная комбинация
                combined_score = 0.7 * distance_score + 0.3 * memory_score
                scores.append((combined_score, j))
            
            next_city = max(scores, key=lambda x: x[0])[1]
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        return tour
    
    def _biased_random_tour(self) -> Tour:
        """
        Генерирует случайный тур с bias к хорошим ребрам из памяти.
        """
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        
        # Применяем небольшую локальную оптимизацию на основе памяти
        for _ in range(min(5, len(tour) // 10)):
            i = random.randint(0, len(tour) - 1)
            j = random.randint(0, len(tour) - 1)
            if i != j:
                # Если swap улучшает память - делаем его
                city_i, city_j = tour[i], tour[j]
                city_i_prev = tour[(i - 1) % len(tour)]
                city_i_next = tour[(i + 1) % len(tour)]
                city_j_prev = tour[(j - 1) % len(tour)]
                city_j_next = tour[(j + 1) % len(tour)]
                
                old_memory = (self.edge_memory[city_i_prev, city_i] + 
                             self.edge_memory[city_i, city_i_next] +
                             self.edge_memory[city_j_prev, city_j] + 
                             self.edge_memory[city_j, city_j_next])
                
                new_memory = (self.edge_memory[city_i_prev, city_j] + 
                             self.edge_memory[city_j, city_i_next] +
                             self.edge_memory[city_j_prev, city_i] + 
                             self.edge_memory[city_i, city_j_next])
                
                if new_memory > old_memory:
                    tour[i], tour[j] = tour[j], tour[i]
        
        return tour

    # ===================== ILS-СЛОЙ НАД ABC =====================

    def _adaptive_local_search(self, tour: Tour) -> Tuple[Tour, float]:
        """
        Выбирает стратегию локального поиска на основе истории улучшений.
        Использует оптимизированный 2-opt с соседями для больших задач.
        """
        tour_array = np.asarray(tour, dtype=np.int32)
        
        # Анализируем историю улучшений
        # ВАЖНО: для больших задач (300+) НЕ используем 3-opt - он слишком медленный (O(n³))
        # Для 400 городов 3-opt = 400³ = 64 млн итераций - это займет минуты!
        use_3opt = False
        if len(self._ls_improvement_history) > 10 and self.num_cities < 200:
            recent_avg = np.mean(self._ls_improvement_history[-5:])
            if recent_avg < 0.1:  # Если последние улучшения слабые
                # Используем более агрессивный 3-opt ТОЛЬКО для малых задач
                use_3opt = True
        
        if use_3opt:
            # print(f"      Используем 3-opt")
            return local_search_3opt(self.distance_matrix, tour)
        
        # Стандартный 2-opt с использованием списков соседей для ускорения
        # Это быстрее и эффективнее для больших задач
        if self.neighbors is not None:
            optimized = fast_2opt_neighbors(self.distance_matrix, tour_array, self.neighbors)
            distance = calculate_tour_length(self.distance_matrix, optimized)
            return list(optimized), float(distance)
        else:
            return local_search_2opt(self.distance_matrix, tour)
    
    def _local_2opt_search_global(self) -> None:
        """
        Периодический вызов адаптивного локального поиска над текущим лучшим маршрутом.
        После улучшения — частично распространяем результат на популяцию (элитизм).
        """
        # print(f"\n    ОТЛАДКА _local_2opt_search_global:")
        # print(f"      Вызов функции начат")
        
        if self.best_tour is None:
            # print(f"       best_tour is None, выход из функции")
            return

        old_distance = self.best_distance
        # print(f"      Текущий best_distance: {old_distance:.2f}")
        # print(f"      Размер тура: {len(self.best_tour)}")
        # print(f"      История улучшений: {len(self._ls_improvement_history)} записей")
        # if len(self._ls_improvement_history) > 0:
            # recent_avg = np.mean(self._ls_improvement_history[-5:]) if len(self._ls_improvement_history) >= 5 else 0
            # print(f"      Среднее последних улучшений: {recent_avg:.4f}")
        
        # print(f"      Вызываем _adaptive_local_search...")
        start_time = time.time()
        
        try:
            improved_tour, improved_dist = self._adaptive_local_search(self.best_tour)
            elapsed = time.time() - start_time


        except Exception as e:
            elapsed = time.time() - start_time
            import traceback
            traceback.print_exc()
            return
        
        # Записываем улучшение в историю
        improvement = old_distance - improved_dist
        self._ls_improvement_history.append(improvement)
        if len(self._ls_improvement_history) > 20:
            self._ls_improvement_history.pop(0)
        
        # print(f"      Улучшение: {old_distance:.2f} -> {improved_dist:.2f} (Δ={improvement:.2f})")

        if improved_dist + 1e-9 < self.best_distance:
            # print(f"      ✅ ГЛОБАЛЬНОЕ УЛУЧШЕНИЕ! {self.best_distance:.2f} -> {improved_dist:.2f}")
            self.best_distance = improved_dist
            self.best_tour = improved_tour.copy()
            
            # Сохраняем в историю лучших решений
            if len(self.historical_best_solutions) >= self.max_historical_solutions:
                self.historical_best_solutions.pop(0)
            self.historical_best_solutions.append(improved_tour.copy())

            # Распространяем улучшенное решение на часть популяции
            num_elite = max(1, len(self.employed_bees) // 5)
            # print(f"      Распространяем на {num_elite} лучших пчел...")
            for i in range(num_elite):
                # небольшое возмущение, чтобы сохранить разнообразие
                if i == 0:
                    new_tour = improved_tour.copy()
                else:
                    new_tour = perturbation_2opt_random(self.distance_matrix, improved_tour, strength=1)

                self.employed_bees[i].solution = new_tour
                self.employed_bees[i].fitness = self._fitness(new_tour)
                self.employed_bees[i].trial = 0
                if hasattr(self.employed_bees[i], 'invalidate_cache'):
                    self.employed_bees[i].invalidate_cache()
            # print(f"      Распространение завершено")
        # else:
            # print(f"      Улучшения нет. Результат ({improved_dist:.2f}) не лучше текущего ({self.best_distance:.2f})")
        
        # print(f"      Функция завершена\n")
    
    def _deep_local_search_global(self) -> None:
        """
        Глубокий локальный поиск с Lin-Kernighan эвристикой.
        Используется для максимальной полировки лучшего решения (элитизм).
        
        Lin-Kernighan - это переменная глубина k-opt поиск, который может
        найти улучшения, которые пропускает обычный 2-opt.
        Согласно исследованиям, может снизить отклонение с 0.5% до 0%.
        """
        if self.best_tour is None:
            return
        # print("Глубокий анализ")
        # Используем упрощенный Lin-Kernighan для глубокого поиска
        # Это более мощный локальный поиск, чем стандартный 2-opt
        improved_tour, improved_dist = lin_kernighan_simplified(
            self.distance_matrix, 
            self.best_tour,
            neighbors=self.neighbors,
            max_depth=3  # Пробуем до 5-opt
        )
        
        if improved_dist + 1e-9 < self.best_distance:
            self.best_distance = improved_dist
            self.best_tour = improved_tour.copy()
            
            # Распространяем на лучшие пчелы
            num_elite = max(1, len(self.employed_bees) // 3)
            for i in range(num_elite):
                if i == 0:
                    self.employed_bees[i].solution = improved_tour.copy()
                else:
                    # Небольшое возмущение
                    perturbed = double_bridge_perturbation(improved_tour)
                    self.employed_bees[i].solution = perturbed
                
                self.employed_bees[i].fitness = self._fitness(self.employed_bees[i].solution)
                self.employed_bees[i].trial = 0
                if hasattr(self.employed_bees[i], 'invalidate_cache'):
                    self.employed_bees[i].invalidate_cache()
    
    def _global_kick(self) -> None:
        """
        Глобальный kick при застревании - делает Double Bridge всем пчелам,
        кроме лучшей. Это позволяет "выпрыгнуть" из глубокого локального минимума.
        """
        if self.best_tour is None:
            # print(" Нет лучшего тура для kick!")
            return
        
        # Находим лучшую пчелу
        best_idx = 0
        best_len = calculate_tour_length(self.distance_matrix, self.employed_bees[0].solution)
        for i in range(1, len(self.employed_bees)):
            current_len = calculate_tour_length(self.distance_matrix, self.employed_bees[i].solution)
            if current_len < best_len:
                best_len = current_len
                best_idx = i
        
        # print(f"   Лучшая пчела: индекс {best_idx}, длина: {best_len:.2f}")
        
        # Для больших задач используем более сильное возмущение
        improved_count = 0
        new_best_found = False
        
        # Делаем Double Bridge всем пчелам, кроме лучшей
        for i in range(len(self.employed_bees)):
            if i != best_idx:
                # Берем лучшее решение и делаем возмущение
                if self.num_cities >= 200:
                    # Двойное возмущение для больших задач
                    candidate = double_bridge_perturbation(self.best_tour)
                    candidate = multi_insert_perturbation(candidate, num_cities_to_move=2)
                else:
                    candidate = double_bridge_perturbation(self.best_tour)
                
                old_len = calculate_tour_length(self.distance_matrix, self.employed_bees[i].solution)
                
                # Применяем быстрый 2-opt для возврата в локальный минимум
                candidate_array = np.asarray(candidate, dtype=np.int32)
                if self.neighbors is not None:
                    optimized = fast_2opt_neighbors(self.distance_matrix, candidate_array, self.neighbors)
                else:
                    optimized = fast_2opt(self.distance_matrix, candidate_array)
                
                new_tour = list(optimized)
                new_len = calculate_tour_length(self.distance_matrix, new_tour)
                
                self.employed_bees[i].solution = new_tour
                self.employed_bees[i].fitness = self._fitness(new_tour)
                self.employed_bees[i].trial = 0
                if hasattr(self.employed_bees[i], 'invalidate_cache'):
                    self.employed_bees[i].invalidate_cache()
                
                # Проверяем улучшение
                if new_len < old_len:
                    improved_count += 1
                
                # Проверяем, не нашли ли лучшее решение
                if new_len < self.best_distance:
                    improvement = self.best_distance - new_len
                    # print(f"Пчела {i}: найдено улучшение! {self.best_distance:.2f} -> {new_len:.2f} (Δ={improvement:.2f})")
                    self.best_distance = new_len
                    self.best_tour = new_tour.copy()
                    new_best_found = True
        
        # print(f"   Результаты kick: {improved_count}/{len(self.employed_bees)-1} пчел улучшились")
        # if not new_best_found:
            # print(f"Новый глобальный минимум не найден")
    
    # ===================== ВИЗУАЛИЗАЦИЯ =====================
    
    def plot_convergence(self, optimal_value: Optional[float] = None) -> None:
        """
        Визуализация сходимости алгоритма.
        Строит график изменения лучшего расстояния по итерациям.
        
        :param optimal_value: Оптимальное значение (опционально, для отображения разницы)
        """
        if len(self.history) == 0:
            print("Нет данных для визуализации (history пуст)")
            return
        
        plt.figure(figsize=(12, 6))

        
        # Если есть оптимальное значение - показываем разницу
        if optimal_value is not None:
            differences = [dist - optimal_value for dist in self.history]
            plt.subplot(1, 2, 2)
            plt.plot(differences, linewidth=2, color='red', label='Разница с оптимумом')
            plt.title("График сходимости", fontsize=14, fontweight='bold')
            plt.xlabel("Итерация", fontsize=12)
            plt.ylabel("Разница с оптимумом", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()

        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence_simple(self) -> None:
        """
        Простая визуализация сходимости (один график).
        """
        if len(self.history) == 0:
            print("Нет данных для визуализации (history пуст)")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, linewidth=2, color='blue')
        plt.title("График сходимости", fontsize=14, fontweight='bold')
        plt.xlabel("Итерация", fontsize=12)
        plt.ylabel("Лучшее расстояние", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()


