from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from generated_matrix import GENERATED_MATRIX

def solve_with_ortools(distance_matrix, timeout=30):
    """Решает TSP через OR-Tools с ограничением по времени"""
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.FromSeconds(timeout)  # Лимит времени

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(route[0])  # Замкнуть маршрут
        
        total_distance = solution.ObjectiveValue()
        return route, total_distance
    
    return None, None

# Сравнение
ortools_route, ortools_distance = solve_with_ortools(GENERATED_MATRIX)
print(f"OR-Tools: Длина = {ortools_distance:.2f}, Маршрут = {ortools_route}")

# Запусти свой алгоритм и сравни:
# your_route, your_distance = ...  # Твой результат
# print(f"ABC: Длина = {your_distance:.2f}, Маршрут = {your_route}")
# print(f"Отклонение: {((your_distance - ortools_distance) / ortools_distance * 100):.2f}%")