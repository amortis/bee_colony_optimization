from random import randint


# Генератор матрицы, к алгоритму отношения не имеет
def generate_matrix(n: int):
    """
    Генератор матрицы
    :param n: Размерность матрицы
    """
    distane_matrix = [
        [randint(1, 100) for i in range(n)] for j in range(n)
    ]
    return distane_matrix


print(generate_matrix(150))
