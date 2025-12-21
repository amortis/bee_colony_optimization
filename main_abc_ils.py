from abc_ils_hybrid import ABCTSPILS
from matrix_task.tsp_task import load_tsplib_instance
import pandas as pd
from typing import Dict


def get_adaptive_parameters(num_cities: int) -> Dict:
    """
    Адаптивно подбирает параметры алгоритма в зависимости от размера задачи.
    
    :param num_cities: Количество городов в задаче TSP
    :return: Словарь с оптимальными параметрами
    """
    if num_cities < 100:
        # Малые задачи (<100 городов)
        return {
            'num_employed_bees': 15,
            'num_onlooker_bees': 20,
            'limit': 100,
            'patience': 150,
            'local_search_interval': 20,
            'heuristic_init_ratio': 0.8,
            'max_iterations': 1000
        }
    elif num_cities < 200:
        # Средние задачи (100-199 городов)
        return {
            'num_employed_bees': 20,
            'num_onlooker_bees': 25,
            'limit': 100,
            'patience': 200,
            'local_search_interval': 25,
            'heuristic_init_ratio': 0.85,
            'max_iterations': 2000
        }
    elif num_cities < 300:
        # Средне-большие задачи (200-299 городов)
        return {
            'num_employed_bees': 15,
            'num_onlooker_bees': 30,
            'limit': 150,
            'patience': 500,
            'local_search_interval': 30,
            'heuristic_init_ratio': 0.9,
            'max_iterations': 3000
        }
    elif num_cities < 500:
        # Большие задачи (300-499 городов)
        return {
            'num_employed_bees': 40,
            'num_onlooker_bees': 50,
            'limit': 60,
            'patience': 350,
            'local_search_interval': 30,
            'heuristic_init_ratio': 0.95,
            'max_iterations': 4000
        }
    else:
        # Очень большие задачи (500+ городов)
        return {
            'num_employed_bees': max(50, num_cities // 10),  # Масштабируем с размером
            'num_onlooker_bees': max(60, num_cities // 8),
            'limit': 50,
            'patience': 400,
            'local_search_interval': 35,
            'heuristic_init_ratio': 0.95,
            'max_iterations': 5000
        }


def main():
    tasks = [
        ("st70.tsp", 675),
        ("rd100.tsp", 7190),
        ("a280.tsp", 2579),
        ("lin318.tsp", 42029),
        ("pa561.tsp", 2763),
        ("rat575.tsp", 6773)
    ]
    
    # Красивый вывод через pandas
    df = pd.DataFrame({
        '№': range(1, len(tasks) + 1),
        'Файл': [task[0] for task in tasks],
        'Оптимальное значение': [task[1] for task in tasks]
    })
    print("\nДоступные задачи TSP:")
    print(df.to_string(index=False))
    print()
    
    user_choice = input("Выберите номер задачи (1-6): ")
    while user_choice not in "123456":
        print("\nНеправильный номер задачи!")
        print(df.to_string(index=False))
        user_choice = input("\nВыберите номер задачи (1-6): ")

    task_name, optimal = tasks[int(user_choice) - 1]
    distance_matrix = load_tsplib_instance(f"matrix_task/{task_name}")
    num_cities = len(distance_matrix)

    print("TSPLIB instance loaded.")
    if optimal is not None:
        print(f"Known optimal value from TSPLIB (or overridden): {optimal}")
    
    # Адаптивный подбор параметров в зависимости от размера задачи
    params = get_adaptive_parameters(num_cities)
    print(f"\nПараметры для задачи с {num_cities} городами:")
    params_df = pd.DataFrame({
        'Параметр': list(params.keys()),
        'Значение': list(params.values())
    })
    print(params_df.to_string(index=False))
    print()
    
    global_history = []
    for _ in range(1):
        abc_ils = ABCTSPILS(
            distance_matrix=distance_matrix,
            num_employed_bees=params['num_employed_bees'],
            num_onlooker_bees=params['num_onlooker_bees'],
            limit=params['limit'],
            patience=params['patience'],
            local_search_interval=params['local_search_interval'],
            heuristic_init_ratio=params['heuristic_init_ratio'],
            use_parallel=True,
            num_workers=16,
            use_gpu=True,
            optimal_value=optimal
        )

        best_tour, best_distance = abc_ils.run(max_iterations=params['max_iterations'])

        print("\n--- FINAL RESULT ---")
        print("Best tour:", best_tour)
        print(f"Best distance: {best_distance:.2f}")
        if optimal is not None:
            gap = best_distance - optimal
            gap_percent = (gap / optimal * 100) if optimal > 0 else 0
            print(f"Gap to optimal: {gap:.2f} ({gap_percent:.2f}%)")

        global_history.append(best_distance)
    print(global_history)


if __name__ == "__main__":
    main()


