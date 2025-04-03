import itertools

# Матрица расстояний
DISTANCE_MATRIX = [
    [0, 29, 82, 46, 68, 52, 72, 42, 51, 55],  # Расстояния от города 0
    [29, 0, 55, 46, 42, 43, 43, 23, 23, 31],  # Расстояния от города 1
    [82, 55, 0, 68, 46, 55, 23, 43, 41, 29],  # Расстояния от города 2
    [46, 46, 68, 0, 82, 15, 72, 31, 62, 42],  # Расстояния от города 3
    [68, 42, 46, 82, 0, 74, 23, 52, 21, 46],  # Расстояния от города 4
    [52, 43, 55, 15, 74, 0, 61, 23, 55, 31],  # Расстояния от города 5
    [72, 43, 23, 72, 23, 61, 0, 42, 23, 31],  # Расстояния от города 6
    [42, 23, 43, 31, 52, 23, 42, 0, 33, 15],  # Расстояния от города 7
    [51, 23, 41, 62, 21, 55, 23, 33, 0, 29],  # Расстояния от города 8
    [55, 31, 29, 42, 46, 31, 31, 15, 29, 0]   # Расстояния от города 9
]

# Функция для вычисления длины маршрута
def calculate_route_length(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += DISTANCE_MATRIX[route[i]][route[i + 1]]
    # Добавляем возврат в начальный город
    total_distance += DISTANCE_MATRIX[route[-1]][route[0]]
    return total_distance

# Полный перебор всех возможных маршрутов
cities = list(range(len(DISTANCE_MATRIX)))
min_distance = float('inf')
optimal_route = None

for perm in itertools.permutations(cities):
    distance = calculate_route_length(perm)
    if distance < min_distance:
        min_distance = distance
        optimal_route = perm

# Вывод результата
print("Оптимальный маршрут:", optimal_route)
print("Минимальная длина маршрута:", min_distance)
# Минимальная длина маршрута: 247