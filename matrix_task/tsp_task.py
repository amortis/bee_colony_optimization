import tsplib95  # pip install tsplib95
import numpy as np

def load_tsplib_instance(filename):
    """Загружает задачу из TSPLIB формата"""
    problem = tsplib95.load(filename)

    
    # Создаём матрицу расстояний
    nodes = list(problem.get_nodes())
    dimension = len(nodes)
    distance_matrix = np.zeros((dimension, dimension))
    
    for i in range(dimension):
        for j in range(dimension):
            distance_matrix[i][j] = problem._wfunc(i+1, j+1)  # type: ignore # TSPLIB индекс с 1
    
    return distance_matrix
