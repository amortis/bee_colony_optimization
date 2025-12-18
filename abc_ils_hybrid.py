import random
import time
from datetime import timedelta
from typing import List, Tuple
import multiprocessing as mp
import concurrent.futures
import warnings

import numpy as np

# Попытка импорта CuPy для GPU, если доступно
# Подавляем предупреждение о CUDA_PATH (не критично, если GPU работает)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    try:
        import cupy as cp
        GPU_AVAILABLE = True
    except ImportError:
        GPU_AVAILABLE = False
        cp = None

from bees import EmployedBee, OnlookerBee
from tsp_optimizations import (
    calculate_tour_length,
    local_search_2opt,
    local_search_2opt_limited,
    local_search_3opt,
    nearest_neighbor_init,
    greedy_init,
    furthest_insertion_init,
    cheapest_insertion_init,
    double_bridge_perturbation,
    perturbation_2opt_random,
    apply_random_mutation,
    tour_similarity,
    apply_crossover,
)


Tour = List[int]


# Функции на уровне модуля для pickle-совместимости
def _generate_solution_worker(args: Tuple[int, np.ndarray, int, int]) -> Tour:
    """Воркер для генерации решения (должен быть на уровне модуля для pickle)."""
    idx, dist_matrix, num_cities, num_heuristic = args
    if idx == 0 and num_heuristic > 0:
        return nearest_neighbor_init(dist_matrix, start_city=0)
    elif idx < num_heuristic:
        return nearest_neighbor_init(dist_matrix, start_city=None)
    elif idx == num_heuristic:
        return greedy_init(dist_matrix)
    else:
        tour = list(range(num_cities))
        random.shuffle(tour)
        return tour


def _compute_fitness_worker(args: Tuple[Tour, np.ndarray]) -> Tuple[Tour, float]:
    """Воркер для вычисления фитнеса (должен быть на уровне модуля для pickle)."""
    sol, dist_matrix = args
    tour_len = calculate_tour_length(dist_matrix, sol)
    return sol, tour_len


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
        num_workers: int | None = None,
        use_gpu: bool = True,
    ):
        # матрица расстояний как numpy float64 (из ILS)
        self.distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        self.num_cities = self.distance_matrix.shape[0]
        
        # GPU поддержка с проверкой работоспособности
        self.use_gpu = False
        self.distance_matrix_gpu = None
        
        if use_gpu and GPU_AVAILABLE:
            try:
                # Пробуем создать тестовый массив на GPU
                test_array = cp.array([1.0, 2.0, 3.0])
                del test_array
                # Пробуем перенести матрицу на GPU
                self.distance_matrix_gpu = cp.asarray(self.distance_matrix)
                self.use_gpu = True
                print(f"GPU acceleration enabled (CuPy)")
            except Exception as e:
                self.use_gpu = False
                self.distance_matrix_gpu = None
                print(f"GPU initialization failed: {type(e).__name__}, using CPU")
        elif use_gpu:
            print("GPU requested but CuPy not available, using CPU")

        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.limit = limit
        self.base_limit = limit  # Сохраняем базовое значение для адаптации
        self.patience = patience
        self.local_search_interval = max(1, local_search_interval)
        self.base_local_search_interval = local_search_interval
        self.heuristic_init_ratio = min(1.0, max(0.0, heuristic_init_ratio))
        
        # Элитный архив лучших уникальных решений
        self.elite_archive: List[Tuple[Tour, float]] = []  # (tour, distance)
        self.elite_archive_size = min(10, num_employed_bees // 5)
        
        # Параметры диверсификации
        self.diversity_threshold = 0.3  # Порог подобия для диверсификации
        self.last_diversity_check = 0
        self.diversity_check_interval = 50

        # Параллельность (отключаем для малых задач - overhead больше выгоды)
        self.num_workers = num_workers
        if self.num_workers and self.num_cities > 200:  # Только для больших задач
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
            self.use_parallel = True
        else:
            self.executor = None
            self.use_parallel = False

        self.employed_bees: List[EmployedBee] = []
        self.onlooker_bees: List[OnlookerBee] = []

        self.best_tour: Tour | None = None
        self.best_distance: float = float("inf")

        self.history: List[float] = []

        self.start_time: float | None = None
        self.end_time: float | None = None
        self.wait = 0
        self.best_iteration = 0

    # ===================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ =====================

    def _calculate_tour_length_gpu(self, tour: Tour) -> float:
        """Вычисление длины тура на GPU (быстрее для больших задач)."""
        if not self.use_gpu or self.distance_matrix_gpu is None:
            return calculate_tour_length(self.distance_matrix, tour)
        
        try:
            tour_gpu = cp.asarray(tour, dtype=cp.int32)
            total = cp.float64(0.0)
            n = len(tour)
            for i in range(n):
                a = tour_gpu[i]
                b = tour_gpu[(i + 1) % n]
                total += self.distance_matrix_gpu[a, b]
            return float(total)
        except Exception:
            # Если GPU ошибка - отключаем GPU и используем CPU
            self.use_gpu = False
            self.distance_matrix_gpu = None
            return calculate_tour_length(self.distance_matrix, tour)
    
    def _calculate_tour_lengths_batch_gpu(self, tours: List[Tour]) -> List[float]:
        """Пакетное вычисление длин туров на GPU (очень эффективно)."""
        if not self.use_gpu or self.distance_matrix_gpu is None or len(tours) < 5:
            return [calculate_tour_length(self.distance_matrix, tour) for tour in tours]
        
        try:
            n = len(tours)
            num_cities = self.num_cities
            tours_gpu = cp.asarray(tours, dtype=cp.int32)
            distances = cp.zeros(n, dtype=cp.float64)
            
            # Векторизованное вычисление на GPU
            for i in range(num_cities):
                a = tours_gpu[:, i]
                b = tours_gpu[:, (i + 1) % num_cities]
                distances += self.distance_matrix_gpu[a, b]
            
            return cp.asnumpy(distances).tolist()
        except Exception:
            # Если GPU ошибка - отключаем GPU и используем CPU
            self.use_gpu = False
            self.distance_matrix_gpu = None
            return [calculate_tour_length(self.distance_matrix, tour) for tour in tours]
    
    def _fitness(self, tour: Tour) -> float:
        """
        Фитнес: чем меньше длина маршрута, тем лучше.
        Используем 1 / distance.
        """
        if self.use_gpu and self.num_cities > 50:
            d = self._calculate_tour_length_gpu(tour)
        else:
            d = calculate_tour_length(self.distance_matrix, tour)
        return 1.0 / (1e-9 + d)
    
    def _validate_and_update_best(self, tour: Tour, distance: float) -> bool:
        """
        Проверяет валидность тура и обновляет best_distance если нужно.
        Возвращает True если обновление произошло.
        """
        # Проверка валидности тура
        if len(tour) != self.num_cities:
            return False
        if set(tour) != set(range(self.num_cities)):
            return False
        
        # Пересчитываем расстояние для проверки
        if self.use_gpu and self.num_cities > 50:
            recalculated = self._calculate_tour_length_gpu(tour)
        else:
            recalculated = calculate_tour_length(self.distance_matrix, tour)
        
        # Проверка что пересчитанное расстояние совпадает с переданным
        if abs(recalculated - distance) > 1e-6:
            print(f"WARNING: Distance mismatch! Passed: {distance:.2f}, Recalculated: {recalculated:.2f}")
            distance = recalculated
        
        # Обновляем если лучше
        if distance + 1e-9 < self.best_distance:
            self.best_distance = distance
            self.best_tour = tour.copy()
            self._update_elite_archive(tour, distance)
            return True
        return False
    
    def _update_elite_archive(self, tour: Tour, distance: float) -> None:
        """Обновляет элитный архив уникальных лучших решений."""
        # Проверяем уникальность (по подобию)
        is_unique = True
        for elite_tour, elite_dist in self.elite_archive:
            if tour_similarity(tour, elite_tour) < 0.1:  # Очень похожие туры
                is_unique = False
                # Обновляем если лучше
                if distance < elite_dist:
                    self.elite_archive.remove((elite_tour, elite_dist))
                    self.elite_archive.append((tour.copy(), distance))
                    self.elite_archive.sort(key=lambda x: x[1])  # Сортируем по расстоянию
                break
        
        if is_unique:
            self.elite_archive.append((tour.copy(), distance))
            self.elite_archive.sort(key=lambda x: x[1])
            # Ограничиваем размер архива
            if len(self.elite_archive) > self.elite_archive_size:
                self.elite_archive = self.elite_archive[:self.elite_archive_size]
    
    def _adapt_parameters(self) -> None:
        """Адаптивно изменяет параметры алгоритма на основе прогресса."""
        # Адаптация limit: уменьшаем если нет улучшений
        if self.no_improvement_count > 100:
            self.limit = max(int(self.base_limit * 0.7), 20)
        elif self.no_improvement_count > 200:
            self.limit = max(int(self.base_limit * 0.5), 10)
        else:
            self.limit = self.base_limit
        
        # Адаптация local_search_interval: увеличиваем частоту если нет улучшений
        if self.no_improvement_count > 50:
            self.local_search_interval = max(5, int(self.base_local_search_interval * 0.5))
        elif self.no_improvement_count > 150:
            self.local_search_interval = max(3, int(self.base_local_search_interval * 0.3))
        else:
            self.local_search_interval = self.base_local_search_interval
    
    def _check_diversity(self) -> float:
        """Проверяет разнообразие популяции. Возвращает среднее подобие."""
        if len(self.employed_bees) < 2:
            return 1.0
        
        similarities = []
        tours = [bee.solution for bee in self.employed_bees]
        
        for i in range(len(tours)):
            for j in range(i + 1, len(tours)):
                sim = tour_similarity(tours[i], tours[j])
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    def _apply_diversification(self) -> None:
        """Применяет сильные возмущения для увеличения разнообразия."""
        diversity = self._check_diversity()
        
        if diversity < self.diversity_threshold:
            # Низкое разнообразие - применяем сильные возмущения
            num_perturbed = max(1, int(len(self.employed_bees) * 0.3))
            indices = random.sample(range(len(self.employed_bees)), num_perturbed)
            
            for idx in indices:
                bee = self.employed_bees[idx]
                # Сильное возмущение
                if random.random() < 0.5:
                    bee.solution = double_bridge_perturbation(bee.solution)
                else:
                    bee.solution = perturbation_2opt_random(self.distance_matrix, bee.solution, strength=5)
                bee.solution = apply_random_mutation(bee.solution, num_mutations=random.randint(2, 4))
                bee.fitness = self._fitness(bee.solution)
                bee.trial = 0

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
        Улучшенная инициализация популяции с разнообразными эвристиками:
        - nearest neighbor (разные стартовые точки)
        - furthest insertion
        - cheapest insertion
        - greedy
        - краткий ILS на некоторых эвристических решениях
        - случайные перестановки
        """
        total = self.num_employed_bees
        num_heuristic = int(total * max(0.9, self.heuristic_init_ratio))
        
        solutions = []
        heuristics_used = 0
        
        # Разнообразные эвристики
        if num_heuristic > 0:
            # Nearest neighbor с разными стартовыми точками
            for i in range(min(3, num_heuristic)):
                sol = nearest_neighbor_init(self.distance_matrix, start_city=i if i < self.num_cities else None)
                # Краткий 2-opt для улучшения
                if self.num_cities > 20:
                    sol, _ = local_search_2opt_limited(self.distance_matrix, sol, max_iterations=5)
                solutions.append(sol)
                heuristics_used += 1
            
            # Furthest insertion
            if heuristics_used < num_heuristic:
                sol = furthest_insertion_init(self.distance_matrix, start_city=None)
                if self.num_cities > 20:
                    sol, _ = local_search_2opt_limited(self.distance_matrix, sol, max_iterations=5)
                solutions.append(sol)
                heuristics_used += 1
            
            # Cheapest insertion
            if heuristics_used < num_heuristic:
                sol = cheapest_insertion_init(self.distance_matrix, start_city=None)
                if self.num_cities > 20:
                    sol, _ = local_search_2opt_limited(self.distance_matrix, sol, max_iterations=5)
                solutions.append(sol)
                heuristics_used += 1
            
            # Greedy
            if heuristics_used < num_heuristic:
                sol = greedy_init(self.distance_matrix)
                if self.num_cities > 20:
                    sol, _ = local_search_2opt_limited(self.distance_matrix, sol, max_iterations=5)
                solutions.append(sol)
                heuristics_used += 1
            
            # Дополнительные nearest neighbor с случайными стартами
            while heuristics_used < num_heuristic:
                sol = nearest_neighbor_init(self.distance_matrix, start_city=None)
                solutions.append(sol)
                heuristics_used += 1
        
        # Случайные решения
        while len(solutions) < total:
            tour = list(range(self.num_cities))
            random.shuffle(tour)
            solutions.append(tour)

        self.employed_bees.clear()
        self.best_tour = None
        self.best_distance = float("inf")
        self.elite_archive.clear()
        self.no_improvement_count = 0

        # Вычисление фитнеса (параллельно только для больших задач)
        if self.use_parallel:
            fitness_args = [(sol, self.distance_matrix) for sol in solutions]
            fitness_results = list(self.executor.map(_compute_fitness_worker, fitness_args))
        else:
            fitness_results = [(sol, calculate_tour_length(self.distance_matrix, sol)) for sol in solutions]

        # Применяем быстрый 2-opt к эвристическим решениям для улучшения старта
        for idx, (sol, tour_len) in enumerate(fitness_results):
            # Применяем легкий 2-opt к первым 30% эвристических решений
            if idx < len(fitness_results) * 0.3 and tour_len < float('inf'):
                # Быстрый 2-opt (ограниченное количество итераций)
                improved_sol, improved_len = local_search_2opt(self.distance_matrix, sol)
                if improved_len < tour_len:
                    sol = improved_sol
                    tour_len = improved_len
            
            bee = EmployedBee(sol, self._fitness)
            bee.fitness = self._fitness(sol)  # Устанавливаем предвычисленный фитнес
            self.employed_bees.append(bee)

            self._validate_and_update_best(sol, tour_len)

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
        self.no_improvement_count = 0  # Счетчик итераций без улучшения для адаптации

        assert self.best_tour is not None

        for it in range(max_iterations):
            old_best = self.best_distance

            # Адаптация параметров
            self._adapt_parameters()
            
            # Проверка диверсификации
            if it - self.last_diversity_check >= self.diversity_check_interval:
                self._apply_diversification()
                self.last_diversity_check = it
            
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()

            # Адаптивная частота локального поиска
            search_freq = self.local_search_interval
            
            # периодический глобальный 2-opt над лучшим решением (слой ILS)
            if it > 0 and it % search_freq == 0:
                self._local_2opt_search_global()
            
            # 3-opt для более глубокого локального поиска (каждые 50 итераций)
            if it > 0 and it % 50 == 0:
                self._local_3opt_search_global()
            
            # Интенсивный локальный поиск в конце (последние 30% итераций)
            if it > max_iterations * 0.7 and it % (search_freq // 2) == 0:
                self._intensive_local_search()

            self.history.append(self.best_distance)

            if self.best_distance + 1e-9 < old_best:
                self.wait = 0
                self.best_iteration = it
                self.no_improvement_count = 0
            else:
                self.wait += 1
                self.no_improvement_count += 1

            if self.wait >= self.patience:
                print(f"\nEarly stopping at iteration {it}, "
                      f"no improvement for {self.patience} iterations.")
                break

            if it % 50 == 0:
                print(
                    f"Iteration {it}: best distance = {self.best_distance:.2f}, "
                    f"time = {self.get_formatted_time()}"
                )

        self.end_time = time.time()
        # Закрываем executor если использовался
        if self.executor:
            self.executor.shutdown(wait=True)
        assert self.best_tour is not None
        print(f"\nFinished. Best distance = {self.best_distance:.2f}, "
              f"time = {self.get_formatted_time()}")
        return self.best_tour, self.best_distance

    # ===================== ФАЗЫ ABC =====================

    def employed_bee_phase(self) -> None:
        """
        Фаза занятых пчёл с увеличенными мутациями.
        Дополнительно: при большом trial для пчелы используем сильное возмущение (double-bridge).
        """
        for bee in self.employed_bees:
            # Адаптивное возмущение - более агрессивное если нет улучшений
            perturbation_threshold = self.limit // 2
            if self.no_improvement_count > 100:
                perturbation_threshold = self.limit // 3  # Чаще возмущаем
            
            if bee.trial > perturbation_threshold:
                # Пробуем несколько вариантов возмущения
                best_perturbed = None
                best_len = float('inf')
                
                for _ in range(2):  # Пробуем 2 варианта возмущения
                    perturbed = double_bridge_perturbation(bee.solution)
                    perturbed = perturbation_2opt_random(self.distance_matrix, perturbed, strength=3)
                    perturbed = apply_random_mutation(perturbed, num_mutations=random.randint(1, 2))
                    
                    if self.use_gpu and self.num_cities > 50:
                        len_pert = self._calculate_tour_length_gpu(perturbed)
                    else:
                        len_pert = calculate_tour_length(self.distance_matrix, perturbed)
                    
                    if len_pert < best_len:
                        best_len = len_pert
                        best_perturbed = perturbed
                
                if best_perturbed:
                    improved = bee.update_solution(best_perturbed)
                    if improved:
                        bee.trial = 0

            improved_flag = bee.explore(self.employed_bees)
            
            # Мутации при улучшении (умеренно)
            if improved_flag:
                # Одна мутация для исследования окрестности
                if random.random() < 0.5:  # 50% вероятность
                    mutated = apply_random_mutation(bee.solution, num_mutations=1)
                    if self.use_gpu and self.num_cities > 50:
                        mutated_len = self._calculate_tour_length_gpu(mutated)
                        current_len = self._calculate_tour_length_gpu(bee.solution)
                    else:
                        mutated_len = calculate_tour_length(self.distance_matrix, mutated)
                        current_len = calculate_tour_length(self.distance_matrix, bee.solution)
                    if mutated_len < current_len:
                        bee.solution = mutated
                        bee.fitness = self._fitness(mutated)
                        bee.trial = 0

            if self.use_gpu and self.num_cities > 50:
                tour_len = self._calculate_tour_length_gpu(bee.solution)
            else:
                tour_len = calculate_tour_length(self.distance_matrix, bee.solution)
            self._validate_and_update_best(bee.solution, tour_len)

            if improved_flag:
                bee.trial = 0
            else:
                bee.trial += 1

    def onlooker_bee_phase(self) -> None:
        """
        Улучшенная фаза пчёл-наблюдателей с использованием элитного архива и кроссоверов.
        """
        if not self.employed_bees:
            return

        self.onlooker_bees.clear()
        solutions = [bee.solution for bee in self.employed_bees]

        # Пакетное вычисление на GPU для всех наблюдателей
        onlooker_solutions = []
        for _ in range(self.num_onlooker_bees):
            # С вероятностью 20% используем кроссовер из элитного архива
            if self.elite_archive and random.random() < 0.2:
                parent1_tour, _ = random.choice(self.elite_archive)
                parent2 = random.choice(solutions)
                crossover_type = random.choice(["ox", "pmx"])
                new_solution = apply_crossover(parent1_tour, parent2, crossover_type)
                # Применяем легкий 2-opt для улучшения
                if self.num_cities > 20:
                    new_solution, _ = local_search_2opt_limited(self.distance_matrix, new_solution, max_iterations=3)
                onlooker = OnlookerBee(new_solution, self._fitness)
            else:
                onlooker = OnlookerBee(random.choice(solutions), self._fitness)
            
            improved = onlooker.explore(solutions)
            
            # Более агрессивные мутации при улучшении
            if improved:
                # 1-2 мутации с вероятностью 60%
                if random.random() < 0.6:
                    num_muts = random.randint(1, 2)
                    mutated = apply_random_mutation(onlooker.solution, num_mutations=num_muts)
                    if self.use_gpu and self.num_cities > 50:
                        mutated_len = self._calculate_tour_length_gpu(mutated)
                        current_len = self._calculate_tour_length_gpu(onlooker.solution)
                    else:
                        mutated_len = calculate_tour_length(self.distance_matrix, mutated)
                        current_len = calculate_tour_length(self.distance_matrix, onlooker.solution)
                    if mutated_len < current_len:
                        onlooker.solution = mutated
                        onlooker.fitness = self._fitness(mutated)
            
            onlooker_solutions.append(onlooker.solution)
            self.onlooker_bees.append(onlooker)
        
        # Пакетное вычисление длин на GPU
        if self.use_gpu and self.num_cities > 50 and len(onlooker_solutions) > 5:
            tour_lengths = self._calculate_tour_lengths_batch_gpu(onlooker_solutions)
        else:
            tour_lengths = [calculate_tour_length(self.distance_matrix, sol) for sol in onlooker_solutions]
        
        for onlooker, tour_len in zip(self.onlooker_bees, tour_lengths):
            self._validate_and_update_best(onlooker.solution, tour_len)

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
        Фаза разведчиков: если пчела слишком долго не улучшалась,
        создаём новое решение.

        Часть новых решений генерируем как возмущение от текущего лучшего тура
        (идея ILS-perturbation / double-bridge), а часть — полностью случайно.
        """
        if self.best_tour is None:
            return

        for i, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                # Вероятность возмущения 0.9 (увеличена)
                if random.random() < 0.9:
                    # Пробуем несколько вариантов возмущения от лучшего тура
                    best_new = None
                    best_len = float('inf')
                    
                    for _ in range(3):  # Пробуем 3 варианта
                        new_tour = double_bridge_perturbation(self.best_tour)
                        new_tour = perturbation_2opt_random(self.distance_matrix, new_tour, strength=3)
                        new_tour = apply_random_mutation(new_tour, num_mutations=random.randint(1, 2))
                        
                        if self.use_gpu and self.num_cities > 50:
                            len_new = self._calculate_tour_length_gpu(new_tour)
                        else:
                            len_new = calculate_tour_length(self.distance_matrix, new_tour)
                        
                        if len_new < best_len:
                            best_len = len_new
                            best_new = new_tour
                    
                    new_tour = best_new if best_new else self.best_tour.copy()
                else:
                    # полностью случайный
                    new_tour = list(range(self.num_cities))
                    random.shuffle(new_tour)

                self.employed_bees[i].solution = new_tour
                self.employed_bees[i].fitness = self._fitness(new_tour)
                self.employed_bees[i].trial = 0

                if self.use_gpu and self.num_cities > 50:
                    tour_len = self._calculate_tour_length_gpu(new_tour)
                else:
                    tour_len = calculate_tour_length(self.distance_matrix, new_tour)
                self._validate_and_update_best(new_tour, tour_len)

    # ===================== ILS-СЛОЙ НАД ABC =====================

    def _local_2opt_search_global(self) -> None:
        """
        Периодический вызов 2-opt над текущим лучшим маршрутом.
        После улучшения — частично распространяем результат на популяцию (элитизм).
        """
        if self.best_tour is None:
            return

        improved_tour, improved_dist = local_search_2opt(self.distance_matrix, self.best_tour)

        if self._validate_and_update_best(improved_tour, improved_dist):
            self.no_improvement_count = 0

            # Распространяем улучшенное решение на часть популяции с мутациями
            num_elite = max(1, len(self.employed_bees) // 3)  # Увеличена доля элиты до 33%
            for i in range(num_elite):
                # возмущение с мутациями для разнообразия
                if i == 0:
                    new_tour = improved_tour.copy()
                else:
                    new_tour = perturbation_2opt_random(self.distance_matrix, improved_tour, strength=2)
                    new_tour = apply_random_mutation(new_tour, num_mutations=random.randint(1, 2))
                self.employed_bees[i].solution = new_tour
                self.employed_bees[i].fitness = self._fitness(new_tour)
                self.employed_bees[i].trial = 0
    
    def _local_3opt_search_global(self) -> None:
        """
        Периодический вызов 3-opt над текущим лучшим маршрутом (более мощный поиск).
        """
        if self.best_tour is None:
            return
        
        # Используем упрощённый 3-opt (безопасная версия)
        improved_tour, improved_dist = local_search_3opt(self.distance_matrix, self.best_tour)
        
        if self._validate_and_update_best(improved_tour, improved_dist):
            self.no_improvement_count = 0


