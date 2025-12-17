import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matrix_task.sparred_tsp_task import load_tsplib_instance, create_sparse_matrix_with_hamiltonian


print("Загрузка TSP задачи...")
full_matrix, optimal = load_tsplib_instance("matrix_task/st70.tsp")

# Создаем неполный граф с гарантированным циклом
print("Создание неполного графа...")
DISTANCE_MATRIX, hamiltonian_cycle = create_sparse_matrix_with_hamiltonian(
    full_matrix, 
    keep_probability=0.3,  # Оставляем 30% рёбер
    seed=42
)
# Ваша матрица расстояний (взять из переменной DISTANCE_MATRIX или sparse_matrix)
# Например: matrix = np.array(DISTANCE_MATRIX)
matrix = np.array(DISTANCE_MATRIX)  # Замените DISTANCE_MATRIX на имя вашей переменной
num_cities = len(matrix)

print("=== Матрица смежности (первые 10x10) ===")
print("(np.inf означает отсутствие ребра)")
for i in range(min(10, num_cities)):
    for j in range(min(10, num_cities)):
        if np.isinf(matrix[i][j]):
            print("  inf", end=" ")
        else:
            print(f"{matrix[i][j]:5.0f}", end=" ")
    print("...")
print("...\n")

# 1. Статистика графа
total_possible = num_cities * (num_cities - 1)
existing_edges = np.sum(~np.isinf(matrix)) // 2  # для неориентированного графа
print(f"Городов: {num_cities}")
print(f"Всего возможных рёбер (без петель): {total_possible}")
print(f"Существующих рёбер в графе: {existing_edges}")
print(f"Плотность графа: {existing_edges/total_possible:.1%}")
print(f"Средняя степень вершины: {2*existing_edges/num_cities:.1f}\n")

# 2. Построение графа
G = nx.Graph()

# Добавляем рёбра (только где расстояние не бесконечно)
for i in range(num_cities):
    for j in range(i + 1, num_cities):  # избегаем дублирования
        if not np.isinf(matrix[i][j]):
            G.add_edge(i, j, weight=matrix[i][j])

# 3. Визуализация
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # позиции вершин
edges = G.edges()

# Рисуем граф
nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edgelist=edges, width=0.5, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title(f"Визуализация неполного графа (n={num_cities}, рёбер={existing_edges})")
plt.axis('off')
plt.tight_layout()
plt.show()

# 4. Проверка связности
if nx.is_connected(G):
    print("✅ Граф связный (между любыми двумя вершинами есть путь).")
else:
    print("⚠️  Граф НЕ связный. Возможно, некоторые города изолированы.")
    components = list(nx.connected_components(G))
    print(f"   Количество компонент связности: {len(components)}")
    print(f"   Размеры компонент: {[len(c) for c in components]}")