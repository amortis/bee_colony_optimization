import numpy as np
from ortools.sat.python import cp_model
from matrix_task.sparred_tsp_task import load_tsplib_instance, create_sparse_matrix_with_hamiltonian


def solve_tsp_with_ortools_cp(distance_matrix):
    """
    Решает TSP на неполном графе с помощью OR-Tools CP-SAT.
    Автоматически обрабатывает диагональ и тип данных.
    """
    import numpy as np
    from ortools.sat.python import cp_model
    
    # 1. КОПИРУЕМ и исправляем матрицу
    n = len(distance_matrix)
    matrix = distance_matrix.copy().astype(float)
    
    # КРИТИЧЕСКИ ВАЖНО: диагональ должна быть 0
    np.fill_diagonal(matrix, 0.0)
    
    # Заменяем оставшиеся inf на очень большое число
    large_value = 10**9
    if np.any(np.isinf(matrix)):
        print(f"Заменяю {np.sum(np.isinf(matrix))} значений inf на {large_value}")
        matrix[np.isinf(matrix)] = large_value
    
    # OR-Tools требует целые числа. Масштабируем.
    if matrix.dtype != int:
        scale_factor = 1000  # Умножаем на 1000 для сохранения 3 знаков после запятой
        matrix = (matrix * scale_factor).astype(int)
    
    # 2. ДАЛЕЕ ИДЁТ ВАША ОРИГИНАЛЬНАЯ ЛОГИКА СОЗДАНИЯ МОДЕЛИ
    model = cp_model.CpModel()
    edge_vars = {}
    edge_list = []
    
    # Создаём переменные только для существующих рёбер (i != j)
    for i in range(n):
        for j in range(i + 1, n):
            # Теперь matrix[i][j] гарантированно содержит число, не inf
            var = model.NewBoolVar(f"x_{i}_{j}")
            edge_vars[(i, j)] = var
            edge_vars[(j, i)] = var
            edge_list.append((i, j, matrix[i][j]))
    
    # Ограничения: каждая вершина имеет степень 2
    for city in range(n):
        incident_vars = []
        for (u, v), var in edge_vars.items():
            if u == city or v == city:
                incident_vars.append(var)
        model.Add(sum(incident_vars) == 2)
    
    # Устранение подциклов (MTZ)
    u = [model.NewIntVar(0, n - 1, f"u_{i}") for i in range(n)]
    for i in range(1, n):
        for j in range(1, n):
            if i != j and (i, j) in edge_vars:
                model.Add(u[i] - u[j] + n * edge_vars[(i, j)] <= n - 1)
    
    # Целевая функция
    objective_terms = []
    for i, j, cost in edge_list:
        objective_terms.append(edge_vars[(i, j)] * cost)
    model.Minimize(sum(objective_terms))
    
    # Решаем
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers = 8
    
    print("Решаю задачу с OR-Tools CP-SAT...")
    status = solver.Solve(model)
    
    # Результаты
    if status == cp_model.OPTIMAL:
        print(f"Статус: ОПТИМАЛЬНОЕ решение найдено")
    elif status == cp_model.FEASIBLE:
        print(f"Статус: ДОПУСТИМОЕ решение найдено")
    else:
        print(f"Статус: Решение не найдено (код: {status})")
        return None, None
    
    # Извлекаем маршрут
    total_distance_scaled = solver.ObjectiveValue()
    # Возвращаем к исходному масштабу
    total_distance = total_distance_scaled / scale_factor if scale_factor != 1 else total_distance_scaled
    
    # Построение маршрута
    route = [0]
    current_city = 0
    
    while len(route) < n:
        for (u, v), var in edge_vars.items():
            if u == current_city and solver.Value(var) == 1 and v not in route:
                route.append(v)
                current_city = v
                break
    
    print(f"Длина маршрута: {total_distance:.2f}")
    print(f"Маршрут (первые 10): {route[:10]}...")
    
    return total_distance, route


# Загружаем TSP задачу
print("Загрузка TSP задачи...")
full_matrix, optimal = load_tsplib_instance("matrix_task/st70.tsp")

# Создаем неполный граф с гарантированным циклом
print("Создание неполного графа...")
DISTANCE_MATRIX, hamiltonian_cycle = create_sparse_matrix_with_hamiltonian(
    full_matrix, 
    keep_probability=1,  # Оставляем 30% рёбер
    seed=42
)

best_dist, best_route = solve_tsp_with_ortools_cp(DISTANCE_MATRIX)
print(best_dist, best_route)

