import random
import time
from datetime import timedelta
from typing import List, Tuple

import numpy as np

from bees import EmployedBee, OnlookerBee
from tsp_optimizations import (
    calculate_tour_length,
    local_search_2opt,
    nearest_neighbor_init,
    greedy_init,
    double_bridge_perturbation,
    perturbation_2opt_random,
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
        Инициализация популяции с использованием ILS-эвристик:
        - часть туров: nearest neighbor (фиксированный старт и случайный старт),
        - часть: greedy,
        - оставшиеся: случайные перестановки.
        """
        total = self.num_employed_bees
        num_heuristic = int(total * self.heuristic_init_ratio)

        solutions: List[Tour] = []

        # 1) Nearest neighbor с фиксированным стартом 0
        if num_heuristic > 0:
            solutions.append(nearest_neighbor_init(self.distance_matrix, start_city=0))

        # 2) Остальные NN с случайным стартом
        while len(solutions) < num_heuristic:
            solutions.append(nearest_neighbor_init(self.distance_matrix, start_city=None))

        # 3) Greedy-эвристика (по крайней мере один раз)
        solutions.append(greedy_init(self.distance_matrix))

        # 4) Остальные — случайные перестановки
        while len(solutions) < total:
            tour = list(range(self.num_cities))
            random.shuffle(tour)
            solutions.append(tour)

        self.employed_bees.clear()
        self.best_tour = None
        self.best_distance = float("inf")

        for sol in solutions:
            bee = EmployedBee(sol, self._fitness)
            self.employed_bees.append(bee)

            tour_len = calculate_tour_length(self.distance_matrix, bee.solution)
            if tour_len < self.best_distance:
                self.best_distance = tour_len
                self.best_tour = bee.solution.copy()

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
            else:
                self.wait += 1

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

    def employed_bee_phase(self) -> None:
        """
        Фаза занятых пчёл.
        Дополнительно: при большом trial для пчелы используем сильное возмущение (double-bridge).
        """
        for bee in self.employed_bees:
            # Если пчела давно не улучшалась — потрясём её маршрут
            # Уменьшили порог с limit//2 до limit//3 для более частого возмущения
            if bee.trial > self.limit // 3:
                perturbed = double_bridge_perturbation(bee.solution)
                # локальный мягкий ILS-подход (не обязательно, но логично)
                perturbed = perturbation_2opt_random(self.distance_matrix, perturbed, strength=2)
                improved = bee.update_solution(perturbed)
                if improved:
                    bee.trial = 0

            improved_flag = bee.explore(self.employed_bees)

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
        Фаза разведчиков: если пчела слишком долго не улучшалась,
        создаём новое решение.

        Часть новых решений генерируем как возмущение от текущего лучшего тура
        (идея ILS-perturbation / double-bridge), а часть — полностью случайно.
        """
        if self.best_tour is None:
            return

        for i, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                # Увеличили вероятность возмущения с 0.5 до 0.8
                if random.random() < 0.8:
                    # возмущение от лучшего тура
                    new_tour = double_bridge_perturbation(self.best_tour)
                    new_tour = perturbation_2opt_random(self.distance_matrix, new_tour, strength=2)
                else:
                    # полностью случайный
                    new_tour = list(range(self.num_cities))
                    random.shuffle(new_tour)

                self.employed_bees[i].solution = new_tour
                self.employed_bees[i].fitness = self._fitness(new_tour)
                self.employed_bees[i].trial = 0

                tour_len = calculate_tour_length(self.distance_matrix, new_tour)
                if tour_len < self.best_distance:
                    self.best_distance = tour_len
                    self.best_tour = new_tour.copy()

    # ===================== ILS-СЛОЙ НАД ABC =====================

    def _local_2opt_search_global(self) -> None:
        """
        Периодический вызов 2-opt над текущим лучшим маршрутом.
        После улучшения — частично распространяем результат на популяцию (элитизм).
        """
        if self.best_tour is None:
            return

        improved_tour, improved_dist = local_search_2opt(self.distance_matrix, self.best_tour)

        if improved_dist + 1e-9 < self.best_distance:
            self.best_distance = improved_dist
            self.best_tour = improved_tour.copy()

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


