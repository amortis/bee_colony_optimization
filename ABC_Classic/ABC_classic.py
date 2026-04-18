# abc_classic.py
import random
import time
import math
from datetime import timedelta
from typing import List, Callable, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


class ABC_Classic:
    """
    Классическая реализация алгоритма искусственной пчелиной колонии (ABC)
    для задачи коммивояжёра (TSP).
    
    Поддерживает инициализацию через fitness_function или distance_matrix.
    """
    
    # ========================================================================
    # Внутренний класс: базовая пчела
    # ========================================================================
    class _Bee:
        def __init__(self, solution: List[int], fitness_func: Callable):
            self.solution = solution.copy()
            self.fitness_func = fitness_func
            self.fitness = fitness_func(solution)
            self.trial = 0

        def _evaluate(self, sol: List[int]) -> float:
            return self.fitness_func(sol)

        def _greedy_update(self, new_sol: List[int]) -> bool:
            new_fit = self._evaluate(new_sol)
            if new_fit > self.fitness:
                self.solution = new_sol
                self.fitness = new_fit
                self.trial = 0
                return True
            self.trial += 1
            return False

    # ========================================================================
    # Внутренний класс: рабочая пчела (Employed Bee)
    # ========================================================================
    class _EmployedBee(_Bee):
        def explore(self, population: List['_Bee']) -> bool:
            candidates = [b for b in population if b is not self]
            if not candidates:
                return False
            partner = random.choice(candidates)
            new_sol = self._crossover_with_partner(partner.solution)
            return self._greedy_update(new_sol)

        def _crossover_with_partner(self, partner_sol: List[int]) -> List[int]:
            """OX-like кроссовер для перестановок"""
            size = len(self.solution)
            new_sol = [-1] * size
            
            start = random.randint(0, size - 2)
            end = random.randint(start + 1, min(start + size // 2, size))
            new_sol[start:end] = self.solution[start:end]
            
            ptr = 0
            for i in range(size):
                if new_sol[i] == -1:
                    while partner_sol[ptr] in new_sol:
                        ptr += 1
                    new_sol[i] = partner_sol[ptr]
                    ptr += 1
            return new_sol

    # ========================================================================
    # Внутренний класс: пчела-наблюдатель (Onlooker Bee)
    # ========================================================================
    class _OnlookerBee(_Bee):
        def explore(self, population: List['_Bee']) -> bool:
            if not population:
                return False
            probs = self._calc_selection_probs(population)
            candidates = random.choices(population, weights=probs, k=3)
            base_bee = max(candidates, key=lambda b: b.fitness)
            new_sol = self._mutate(base_bee.solution)
            return self._greedy_update(new_sol)

        def _calc_selection_probs(self, population: List['_Bee']) -> List[float]:
            fitnesses = [b.fitness for b in population]
            if not fitnesses:
                return [1/len(population)] * len(population)
            max_f = max(fitnesses)
            exp_vals = [math.exp((f - max_f) * 10) for f in fitnesses]
            total = sum(exp_vals)
            return [e/total for e in exp_vals] if total > 0 else [1/len(population)] * len(population)

        def _mutate(self, base_sol: List[int]) -> List[int]:
            new_sol = base_sol.copy()
            n = len(new_sol)
            if n < 2:
                return new_sol
            mutation = random.choice(['inversion', 'swap', 'shift'])
            if mutation == 'inversion':
                i, j = sorted(random.sample(range(n), 2))
                new_sol[i:j+1] = reversed(new_sol[i:j+1])
            elif mutation == 'swap':
                i, j = random.sample(range(n), 2)
                new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
            else:
                idx = random.randint(0, n-1)
                city = new_sol.pop(idx)
                new_pos = random.randint(0, len(new_sol))
                new_sol.insert(new_pos, city)
            return new_sol

    # ========================================================================
    # Внутренний класс: пчела-разведчик (Scout Bee)
    # ========================================================================
    class _ScoutBee(_Bee):
        def __init__(self, solution: List[int], fitness_func: Callable, lb: int, ub: int):
            super().__init__(solution, fitness_func)
            self.lb = lb
            self.ub = ub

        def explore(self, _) -> bool:
            new_sol = self._generate_random()
            return self._greedy_update(new_sol)

        def _generate_random(self) -> List[int]:
            return random.sample(range(self.lb, self.ub + 1), self.ub - self.lb + 1)

    # ========================================================================
    # Основной класс ABC_Classic
    # ========================================================================
    
    def __init__(
        self,
        fitness_function: Optional[Callable[[List[int]], float]] = None,
        distance_matrix: Optional[np.ndarray] = None,
        lb: Optional[int] = None,
        ub: Optional[int] = None,
        num_employed_bees: int = 50,
        num_onlooker_bees: int = 50,
        limit: int = 20,
        patience: int = 100,
        seed: Optional[int] = None
    ):
        """
        Инициализация алгоритма.
        
        :param fitness_function: функция оценки (чем больше — тем лучше)
        :param distance_matrix: матрица расстояний (альтернатива fitness_function)
        :param lb/ub: границы нумерации городов (авто-определяются из distance_matrix)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Авто-генерация фитнес-функции из distance_matrix
        if distance_matrix is not None:
            self.distance_matrix = np.array(distance_matrix)
            n = len(self.distance_matrix)
            lb = lb if lb is not None else 0
            ub = ub if ub is not None else n - 1
            
            def _tsp_fitness(route: List[int]) -> float:
                total = sum(self.distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
                total += self.distance_matrix[route[-1]][route[0]]
                return 1.0 / total if total > 0 else float('inf')
            
            self.fitness_function = _tsp_fitness
            self._distance_eval = lambda route: sum(self.distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1)) + self.distance_matrix[route[-1]][route[0]]
        elif fitness_function is not None:
            self.fitness_function = fitness_function
            self._distance_eval = None  # пользователь сам знает, как считать расстояние
            self.distance_matrix = None
        else:
            raise ValueError("Требуется либо fitness_function, либо distance_matrix")
            
        self.lb = lb
        self.ub = ub
        self.n_cities = ub - lb + 1 if lb is not None and ub is not None else 0
        
        self.num_employed = num_employed_bees
        self.num_onlooker = num_onlooker_bees
        self.limit = limit
        self.patience = patience
        
        self.employed_bees: List[self._EmployedBee] = []
        self.best_solution: Optional[List[int]] = None
        self.best_fitness = float('-inf')
        
        self._convergence_history: List[float] = []
        self.wait = 0
        self.best_iteration = 0
        self.start_time = None

    def _generate_random_tour(self) -> List[int]:
        return random.sample(range(self.lb, self.ub + 1), self.n_cities)

    def _evaluate_distance(self, route: List[int]) -> float:
        """Вычисляет длину маршрута (если известна distance_matrix)"""
        if self._distance_eval:
            return self._distance_eval(route)
        # fallback: если нет матрицы, возвращаем 1/fitness
        return 1.0 / self.fitness_function(route) if self.fitness_function(route) > 0 else float('inf')

    def _initialize_population(self):
        for _ in range(self.num_employed):
            sol = self._generate_random_tour()
            bee = self._EmployedBee(sol, self.fitness_function)
            self.employed_bees.append(bee)
            if bee.fitness > self.best_fitness:
                self.best_fitness = bee.fitness
                self.best_solution = bee.solution.copy()

    def _employed_bee_phase(self):
        for bee in self.employed_bees:
            bee.explore(self.employed_bees)
            if bee.fitness > self.best_fitness:
                self.best_fitness = bee.fitness
                self.best_solution = bee.solution.copy()

    def _onlooker_bee_phase(self):
        for _ in range(self.num_onlooker):
            base_sol = random.choice(self.employed_bees).solution
            bee = self._OnlookerBee(base_sol, self.fitness_function)
            bee.explore(self.employed_bees)
            if bee.fitness > self.best_fitness:
                self.best_fitness = bee.fitness
                self.best_solution = bee.solution.copy()

    def _scout_bee_phase(self):
        for i, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                new_sol = self._generate_random_tour()
                new_bee = self._EmployedBee(new_sol, self.fitness_function)
                self.employed_bees[i] = new_bee
                if new_bee.fitness > self.best_fitness:
                    self.best_fitness = new_bee.fitness
                    self.best_solution = new_bee.solution.copy()

    def run(
        self,
        max_iterations: int,
        verbose: bool = True,
        log_interval: int = 10
    ) -> Tuple[List[int], float]:
        """
        Запуск алгоритма.
        
        :return: (лучший маршрут, длина маршрута)  # ← возвращаем расстояние, а не фитнес!
        """
        self._initialize_population()
        self.start_time = time.time()
        self._convergence_history = []
        
        for iteration in range(max_iterations):
            old_best = self.best_fitness
            
            self._employed_bee_phase()
            self._onlooker_bee_phase()
            self._scout_bee_phase()
            
            # Сохраняем длину маршрута для истории
            current_distance = self._evaluate_distance(self.best_solution) if self.best_solution else float('inf')
            self._convergence_history.append(current_distance)
            
            if self.best_fitness > old_best:
                self.wait = 0
                self.best_iteration = iteration
            else:
                self.wait += 1
                
            if self.wait >= self.patience:
                if verbose:
                    print(f"\n⚡ Ранняя остановка на итерации {iteration}")
                break
                
            if verbose and iteration % log_interval == 0:
                elapsed = str(timedelta(seconds=time.time() - self.start_time)).split('.')[0]
                print(f"Итерация {iteration:4d} | Время: {elapsed} | Длина: {current_distance:.2f}")
        
        best_distance = self._evaluate_distance(self.best_solution) if self.best_solution else float('inf')
        return self.best_solution, best_distance

    def get_convergence_history(self) -> List[float]:
        """Возвращает историю длин маршрутов по итерациям"""
        return self._convergence_history.copy()

    def plot_convergence(self, optimal_value: Optional[float] = None, title: str = "Сходимость ABC"):
        if not self._convergence_history:
            print("⚠ История пуста — сначала запустите run()")
            return
        plt.figure(figsize=(10, 5))
        iterations = list(range(len(self._convergence_history)))
        if optimal_value is not None:
            diff = [abs(d - optimal_value) for d in self._convergence_history]
            plt.plot(iterations, diff, linewidth=2, label='Ошибка')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            plt.ylabel('Абсолютная ошибка')
        else:
            plt.plot(iterations, self._convergence_history, linewidth=2)
            plt.ylabel('Длина маршрута')
        plt.xlabel('Итерация')
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            return 0
        return time.time() - self.start_time