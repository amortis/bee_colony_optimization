from random import randint


# Генератор матрицы, к алгоритму отношения не имеет
def generate_matrix(n: int) -> list[list[int]]:
    """
    Генерирует симметричную матрицу расстояний с нулями на диагонали

    :param n: Размерность матрицы (количество городов/точек)
    :return: Квадратная матрица n x n, где:
             - diagonal = 0 (расстояние до себя)
             - matrix[i][j] == matrix[j][i] (симметричность)
             - значения от 1 до 100 (случайные расстояния)
    """
    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            value = randint(1, 100)
            matrix[i][j] = value
            matrix[j][i] = value  # Симметричное значение

    return matrix


print(generate_matrix(30))
