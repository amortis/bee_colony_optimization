import random
import time
from datetime import timedelta
import numpy as np
from bees import EmployedBee, OnlookerBee


class ABCAlgorithmSimple:
    """
    Упрощенная и эффективная версия ABC алгоритма для TSP.
    Сохраняет ключевую логику ABC (три фазы пчел), но использует эффективные методы из ILS.
    """
    
    def __init__(self, fitness_function, lb, ub, num_employed_bees, num_onlooker_bees, 
                 limit, patience, optimal_length, distance_matrix=None, 
                 local_search_interval=10, elitism_rate=0.1, heuristic_init_ratio=0.8):
        """
        Инициализация упрощенного ABC алгоритма.
        
        :param fitness_function: Функция оценки качества решения
        :param lb: Нижняя граница (обычно 0)
        :param ub: Верхняя граница (количество городов - 1)
        :param num_employed_bees: Количество рабочих пчел
        :param num_onlooker_bees: Количество пчел-наблюдателей
        :param limit: Лимит неудач перед заменой решения
        :param patience: Ранняя остановка после N итераций без улучшения
        :param optimal_length: Оптимальное значение для сравнения
        :param distance_matrix: Матрица расстояний для эвристик
        :param local_search_interval: Как часто применять эффективный 2-opt
        :param elitism_rate: Доля худших пчел для замены лучшим решением
        :param heuristic_init_ratio: Доля эвристически инициализированных решений
        """
        assert ub > lb, "Верхняя граница должна быть больше нижней"
        assert num_employed_bees > 0, "Должна быть хотя бы одна рабочая пчела"
        assert num_onlooker_bees > 0, "Должна быть хотя бы одна пчела-наблюдатель"
        assert limit > 0, "Лимит неудач должен быть положительным"
        
        self.fitness_function = fitness_function
        self.lb = lb
        self.ub = ub
        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.limit = limit
        self.optimal_length = optimal_length
        self.distance_matrix = distance_matrix
        self.local_search_interval = local_search_interval
        self.elitism_rate = elitism_rate
        self.heuristic_init_ratio = heuristic_init_ratio
        
        # Популяция
        self.employed_bees = []
        self.onlooker_bees = []
        
        # Лучшее решение
        self.best_solution = None
        self.best_fitness = float('-inf')
        
        # История
        self.global_history = []
        self.patience = patience
        self.wait = 0
        self.best_iteration = 0
        
        # Время
        self.start_time = None
    
    def run_algorithm(self, max_iterations):
        """Основной цикл ABC алгоритма."""
        self._initialize_population()
        self.start_time = time.time()
        
        for iteration in range(max_iterations):
            old_best = self.best_fitness
            
            # Три ключевые фазы ABC
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            
            # Эффективный локальный поиск (из ILS)
            if iteration > 0 and iteration % self.local_search_interval == 0:
                self._local_2opt_search()
            
            # Элитизм
            if self.elitism_rate > 0 and iteration % 10 == 0:
                self._apply_elitism()
            
            # История
            if self.best_fitness > 0:
                current_distance = 1 / self.best_fitness
                gap = current_distance - self.optimal_length
                self.global_history.append(gap)
            
            # Проверка улучшения
            if self.best_fitness > old_best:
                self.wait = 0
                self.best_iteration = iteration
            else:
                self.wait += 1
            
            # Ранняя остановка
            if self.wait >= self.patience:
                print(f"\nEarly stopping at iteration {iteration}")
                break
            
            # Прогресс
            if iteration % 20 == 0:
                elapsed = self.get_formatted_time()
                current_distance = 1 / self.best_fitness if self.best_fitness > 0 else float('inf')
                gap = current_distance - self.optimal_length
                print(f"Iter {iteration}. Time: {elapsed}. Distance: {current_distance:.2f}. Gap: {gap:.2f}")
        
        return self.best_solution, self.best_fitness
    
    def _initialize_population(self):
        """Инициализация популяции с использованием nearest neighbor и случайных решений."""
        initial_solutions = []
        
        num_heuristic = int(self.num_employed_bees * self.heuristic_init_ratio)
        num_random = self.num_employed_bees - num_heuristic
        
        # Эвристическая инициализация (nearest neighbor)
        if self.distance_matrix is not None:
            for i in range(num_heuristic):
                solution = self._nearest_neighbor_init(random_start=(i % 2 == 1))
                initial_solutions.append(solution)
        
        # Случайная инициализация
        for _ in range(num_random):
            solution = list(range(self.lb, self.ub + 1))
            random.shuffle(solution)
            initial_solutions.append(solution)
        
        # Создание пчел
        for solution in initial_solutions:
            fitness = self.fitness_function(solution)
            employed_bee = EmployedBee(solution, self.fitness_function, initial_fitness=fitness)
            self.employed_bees.append(employed_bee)
            
            if fitness > self.best_fitness:
                self.best_solution = solution.copy()
                self.best_fitness = fitness
    
    def _nearest_neighbor_init(self, random_start=False):
        """Эвристика ближайшего соседа для инициализации."""
        if self.distance_matrix is None:
            solution = list(range(self.lb, self.ub + 1))
            random.shuffle(solution)
            return solution
        
        unvisited = set(range(self.lb, self.ub + 1))
        tour = []
        
        current = random.choice(list(unvisited)) if random_start else self.lb
        tour.append(current)
        unvisited.remove(current)
        
        while unvisited:
            nearest = min(unvisited, key=lambda city: self.distance_matrix[current][city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return tour
    
    def employed_bee_phase(self):
        """Фаза рабочих пчел: каждая пчела пытается улучшить свое решение."""
        # EmployedBee.explore ожидает список пчел (не решений), так как он использует partner.solution
        for bee in self.employed_bees:
            # Пчела исследует окрестность своего решения
            is_improved = bee.explore(self.employed_bees)
            
            # Обновляем лучшее решение
            if is_improved and bee.fitness > self.best_fitness:
                self.best_solution = bee.solution.copy()
                self.best_fitness = bee.fitness
    
    def onlooker_bee_phase(self):
        """Фаза пчел-наблюдателей: выбирают решения по вероятности и пытаются улучшить."""
        if not self.employed_bees:
            return
        
        self.onlooker_bees.clear()
        
        # Вычисляем вероятности выбора (roulette wheel)
        fitnesses = [bee.fitness for bee in self.employed_bees]
        total_fitness = sum(fitnesses)
        
        if total_fitness > 0:
            probabilities = [f / total_fitness for f in fitnesses]
        else:
            probabilities = [1.0 / len(self.employed_bees)] * len(self.employed_bees)
        
        # Создаем список решений для передачи в explore
        employed_solutions = [bee.solution for bee in self.employed_bees]
        
        # Создаем пчел-наблюдателей
        for _ in range(self.num_onlooker_bees):
            # Выбираем решение на основе вероятности
            selected_bee = random.choices(self.employed_bees, weights=probabilities, k=1)[0]
            
            # Пчела-наблюдатель пытается улучшить выбранное решение
            onlooker = OnlookerBee(selected_bee.solution.copy(), self.fitness_function, 
                                  initial_fitness=selected_bee.fitness)
            # explore ожидает список решений, не список пчел
            is_improved = onlooker.explore(employed_solutions)
            self.onlooker_bees.append(onlooker)
            
            # Обновляем лучшее решение
            if is_improved and onlooker.fitness > self.best_fitness:
                self.best_solution = onlooker.solution.copy()
                self.best_fitness = onlooker.fitness
        
        # Заменяем худшие решения лучшими из наблюдателей
        self._upgrade_solutions()
    
    def _upgrade_solutions(self):
        """Заменяет худшие решения рабочих пчел лучшими решениями наблюдателей."""
        if not self.onlooker_bees:
            return
        
        # Находим лучшее решение среди наблюдателей
        best_onlooker = max(self.onlooker_bees, key=lambda b: b.fitness)
        
        # Находим худшую рабочую пчелу
        worst_employed_idx = min(range(len(self.employed_bees)), 
                                key=lambda i: self.employed_bees[i].fitness)
        
        # Заменяем, если лучше
        if best_onlooker.fitness > self.employed_bees[worst_employed_idx].fitness:
            self.employed_bees[worst_employed_idx].solution = best_onlooker.solution.copy()
            self.employed_bees[worst_employed_idx].fitness = best_onlooker.fitness
            self.employed_bees[worst_employed_idx].trial = 0
    
    def scout_bee_phase(self):
        """Фаза разведчиков: заменяет решения, которые не улучшались долго."""
        for i, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                # Генерируем новое случайное решение
                new_solution = list(range(self.lb, self.ub + 1))
                random.shuffle(new_solution)
                new_fitness = self.fitness_function(new_solution)
                
                bee.solution = new_solution
                bee.fitness = new_fitness
                bee.trial = 0
                
                # Обновляем лучшее решение
                if new_fitness > self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness
    
    def _local_2opt_search(self):
        """
        Эффективный локальный поиск 2-opt (из ILS).
        Находит локальный оптимум для лучшего решения.
        """
        if self.best_solution is None or self.distance_matrix is None:
            return
        
        current_solution = self.best_solution.copy()
        n = len(current_solution)
        
        while True:
            best_improvement = 0
            best_i, best_k = -1, -1
            
            # Ищем лучшее улучшение среди всех возможных 2-opt обменов
            for i in range(n - 1):
                for k in range(i + 1, n):
                    # Учитываем цикличность
                    i_prev = (i - 1) % n
                    k_next = (k + 1) % n
                    
                    # Города
                    A = current_solution[i_prev]
                    B = current_solution[i]
                    C = current_solution[k]
                    D = current_solution[k_next]
                    
                    # Избегаем обмена соседних узлов
                    if i_prev == k or i == k_next:
                        continue
                    
                    # Вычисляем изменение расстояния
                    old_dist = self.distance_matrix[A][B] + self.distance_matrix[C][D]
                    new_dist = self.distance_matrix[A][C] + self.distance_matrix[B][D]
                    improvement = old_dist - new_dist
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i, best_k = i, k
            
            # Применяем лучшее улучшение
            if best_improvement > 0:
                current_solution[best_i:best_k+1] = list(reversed(current_solution[best_i:best_k+1]))
            else:
                # Локальный оптимум достигнут
                break
        
        # Обновляем лучшее решение
        new_fitness = self.fitness_function(current_solution)
        if new_fitness > self.best_fitness:
            self.best_solution = current_solution.copy()
            self.best_fitness = new_fitness
    
    def _apply_elitism(self):
        """Заменяет часть худших решений лучшим решением."""
        if self.best_solution is None or self.elitism_rate <= 0:
            return
        
        num_to_replace = max(1, int(self.elitism_rate * len(self.employed_bees)))
        worst_bees = sorted(self.employed_bees, key=lambda b: b.fitness)[:num_to_replace]
        
        for bee in worst_bees:
            bee.solution = self.best_solution.copy()
            bee.fitness = self.best_fitness
            bee.trial = 0
    
    def get_formatted_time(self, seconds=None):
        """Форматирует время в читаемый вид."""
        if seconds is None:
            seconds = time.time() - self.start_time if self.start_time else 0
        return str(timedelta(seconds=int(seconds))).split(".")[0]
    
    def get_elapsed_time(self):
        """Возвращает прошедшее время."""
        return time.time() - self.start_time if self.start_time else 0
