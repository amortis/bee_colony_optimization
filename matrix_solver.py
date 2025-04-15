from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from distane_matrix import DISTANCE_MATRIX


def solve_tsp_with_ortools(distance_matrix):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        path = []
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        path.append(0)
        return path, solution.ObjectiveValue()
    return None, None

print(solve_tsp_with_ortools(DISTANCE_MATRIX))

import numpy as np
import random


def aco_tsp(distance_matrix, n_ants=30, n_iterations=100, alpha=1, beta=3, evaporation=0.5, Q=100):
    n = len(distance_matrix)
    pheromone = np.ones((n, n))  # Изначально феромоны равны 1

    best_path = None
    best_distance = float('inf')

    for _ in range(n_iterations):
        paths = []
        distances = []

        # Каждый муравей строит путь
        for _ in range(n_ants):
            visited = [False] * n
            path = []
            current = random.randint(0, n - 1)
            path.append(current)
            visited[current] = True

            while len(path) < n:
                unvisited = [i for i in range(n) if not visited[i]]
                probabilities = []

                # Вероятности перехода в каждый город
                for city in unvisited:
                    tau = pheromone[current][city] ** alpha
                    eta = (1 / (distance_matrix[current][city] + 1e-10)) ** beta  # +1e-10 чтобы избежать деления на 0
                    probabilities.append(tau * eta)

                # Нормировка вероятностей
                prob_sum = sum(probabilities)
                probabilities = [p / prob_sum for p in probabilities]

                # Выбор следующего города (рулеточный выбор)
                next_city = random.choices(unvisited, weights=probabilities, k=1)[0]
                path.append(next_city)
                visited[next_city] = True
                current = next_city

            # Возвращаемся в начальный город
            path.append(path[0])
            total_distance = sum(distance_matrix[path[i]][path[i + 1]] for i in range(n))

            paths.append(path)
            distances.append(total_distance)

            # Обновляем лучший путь
            if total_distance < best_distance:
                best_distance = total_distance
                best_path = path

        # Испарение феромонов
        pheromone *= (1 - evaporation)

        # Обновление феромонов на основе пройденных путей
        for path, distance in zip(paths, distances):
            for i in range(n):
                pheromone[path[i]][path[i + 1]] += Q / distance
                pheromone[path[i + 1]][path[i]] += Q / distance  # Если матрица симметрична

    return best_path, best_distance


# Запуск алгоритма
best_path, best_distance = aco_tsp(DISTANCE_MATRIX, n_ants=50, n_iterations=200)
print("Лучший путь:", best_path)
print("Длина пути:", best_distance)

path = [19, 17, 26, 7, 0, 16, 24, 6, 23, 10, 18, 4, 13, 9, 27, 8, 28, 11, 3, 1, 20, 15, 5, 21, 12, 14, 2, 25, 22, 29, 19]
total_distance = 0

for i in range(len(path) - 1):
    total_distance += DISTANCE_MATRIX[path[i]][path[i+1]]

print("Длина маршрута:", total_distance)
