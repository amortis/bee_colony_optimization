import random
import time
from datetime import timedelta
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from bees import EmployedBee, OnlookerBee
from tsp_optimizations import (
    calculate_tour_length,
    local_search_2opt,
    local_search_3opt,
    fast_2opt,
    nearest_neighbor_init,
    greedy_init,
    double_bridge_perturbation,
    perturbation_2opt_random,
    build_candidate_lists,
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
    ):
        # матрица расстояний как numpy float64 (из ILS)
        self.distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        self.num_cities = self.distance_matrix.shape[0]

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
        
        # Candidate Lists для ускорения 2-opt
        self.candidate_lists: Optional[np.ndarray] = None
        if self.num_cities > 50:  # Строим только для больших задач
            self.candidate_lists = build_candidate_lists(self.distance_matrix, num_neighbors=20)

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

        for it in range(max_iterations):
            old_best = self.best_distance

            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()

            # периодический глобальный 2-opt над лучшим решением (слой ILS)
            if it > 0 and it % self.local_search_interval == 0:
                self._local_2opt_search_global()

            self.history.append(self.best_distance)

            if self.best_distance + 1e-9 < old_best:
                self.wait = 0
                self.best_iteration = it
                # Обновляем память о хороших ребрах
                if self.best_tour is not None:
                    self._update_edge_memory(self.best_tour)
            else:
                self.wait += 1

            # Адаптация параметров
            if it > 0:
                self._adapt_parameters(it, max_iterations)
            
            # Adaptive Diversity Control
            if it > 0 and it % 20 == 0:
                self._adaptive_diversity_control()
            
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
        assert self.best_tour is not None
        print(f"\nFinished. Best distance = {self.best_distance:.2f}, "
              f"time = {self.get_formatted_time()}")
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

                # Применяем быстрый локальный поиск к успешным кандидатам (20% вероятность)
                # Это значительно улучшает качество решений
                if improved_flag and random.random() < 0.2:
                    tour_array = np.asarray(bee.solution, dtype=np.int32)
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
                candidate = double_bridge_perturbation(self.best_tour)
                
                # Сразу применяем быстрый 2-opt, чтобы вернуть его в локальный минимум
                # (Идея ILS: Local Opt -> Perturb -> Local Opt)
                candidate_array = np.asarray(candidate, dtype=np.int32)
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
        
        # Если популяция сходится (низкая дисперсия)
        if fitness_variance < self.best_distance * 0.01 or fitness_variance < 1e-6:
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
        
        num_to_perturb = max(1, len(self.employed_bees) // 4)
        for i in range(num_to_perturb):
            idx = (self.best_iteration + i) % len(self.employed_bees)
            if random.random() < 0.7:  # 70% вероятность сильного возмущения
                self.employed_bees[idx].solution = double_bridge_perturbation(self.best_tour)
            else:
                self.employed_bees[idx].solution = self._random_restart()
            
            self.employed_bees[idx].fitness = self._fitness(self.employed_bees[idx].solution)
            self.employed_bees[idx].trial = 0
            self.employed_bees[idx].invalidate_cache()
    
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
        """
        # Анализируем историю улучшений
        if len(self._ls_improvement_history) > 10:
            recent_avg = np.mean(self._ls_improvement_history[-5:])
            if recent_avg < 0.1:  # Если последние улучшения слабые
                # Используем более агрессивный 3-opt
                return local_search_3opt(self.distance_matrix, tour)
        
        # Стандартный 2-opt
        return local_search_2opt(self.distance_matrix, tour)
    
    def _local_2opt_search_global(self) -> None:
        """
        Периодический вызов адаптивного локального поиска над текущим лучшим маршрутом.
        После улучшения — частично распространяем результат на популяцию (элитизм).
        """
        if self.best_tour is None:
            return

        old_distance = self.best_distance
        improved_tour, improved_dist = self._adaptive_local_search(self.best_tour)
        
        # Записываем улучшение в историю
        improvement = old_distance - improved_dist
        self._ls_improvement_history.append(improvement)
        if len(self._ls_improvement_history) > 20:
            self._ls_improvement_history.pop(0)

        if improved_dist + 1e-9 < self.best_distance:
            self.best_distance = improved_dist
            self.best_tour = improved_tour.copy()
            
            # Сохраняем в историю лучших решений
            if len(self.historical_best_solutions) >= self.max_historical_solutions:
                self.historical_best_solutions.pop(0)
            self.historical_best_solutions.append(improved_tour.copy())

            # Распространяем улучшенное решение на часть популяции
            num_elite = max(1, len(self.employed_bees) // 5)
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


