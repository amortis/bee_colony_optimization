from random import randint
import os

def generate_and_save_matrix(n: int, filename: str = "generated_matrix.py"):
    """
    Генерирует матрицу расстояний и сохраняет её в Python-файл для импорта
    
    :param n: Размерность матрицы
    :param filename: Имя выходного файла (по умолчанию: generated_matrix.py)
    """
    # Генерация симметричной матрицы с нулями на диагонали
    generated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = randint(10, 100)  # Диапазон расстояний
            generated_matrix[i][j] = distance
            generated_matrix[j][i] = distance
    
    # Сохранение в файл с правильным форматированием
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Автоматически сгенерированная матрица расстояний\n")
        f.write(f"# Размерность: {n}x{n}\n")
        f.write(f"# Генерация выполнена при помощи скрипта generate_matrix.py\n\n")
        f.write("GENERATED_MATRIX = [\n")
        
        for i, row in enumerate(generated_matrix):
            # Форматируем строку для красивого вывода
            row_str = ", ".join(f"{val:3d}" for val in row)
            f.write(f"    [{row_str}],\n")
        
        f.write("]\n")
    
    print(f"✅ Матрица размером {n}x{n} успешно сохранена в {os.path.abspath(filename)}")

# Запуск генерации при выполнении скрипта
if __name__ == "__main__":
    generate_and_save_matrix(70)