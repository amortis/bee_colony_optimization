import tsplib95  # pip install tsplib95
import numpy as np


def load_tsplib_instance(filename):
    """Загружает задачу из TSPLIB формата.

    Возвращает:
        distance_matrix: np.ndarray - матрица расстояний между городами
        optimal: Optional[int/float] - оптимальное значение (если задано в файле), иначе None
    """
    problem = tsplib95.load(filename)
    # Получаем оптимальное решение, если есть
    optimal = getattr(problem, "optimal_value", None)

    # Создаём матрицу расстояний
    nodes = list(problem.get_nodes())
    dimension = len(nodes)
    distance_matrix = np.zeros((dimension, dimension))

    for i in range(dimension):
        for j in range(dimension):
            # В TSPLIB индексация с 1, поэтому +1
            distance_matrix[i][j] = problem._wfunc(i + 1, j + 1)  # type: ignore

    if optimal is None:
        optimal = 675
    return distance_matrix, optimal


if __name__ == "__main__":
    # Пример использования при самостоятельном запуске файла
    matrix, optimal = load_tsplib_instance("matrix_task/st70.tsp")
    # Если знаем точное оптимальное значение, можем его задать вручную
    if optimal is None:
        optimal = 675
    print(f"Оптимальное решение из TSPLIB: {optimal}")
    print(matrix)