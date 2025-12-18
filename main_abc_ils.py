from abc_ils_hybrid import ABCTSPILS
from matrix_task.tsp_task import load_tsplib_instance


def main():
    # Загружаем задачу TSP из TSPLIB-файла
    # Можно менять путь на eil51.tsp, st70.tsp и т.д.
    distance_matrix, optimal = load_tsplib_instance("matrix_task/rd100.tsp")

    print("TSPLIB instance loaded.")
    if optimal is not None:
        print(f"Known optimal value from TSPLIB (or overridden): {optimal}")
    global_history = []
    for _ in range(1):
        # Создаём гибридный ABC+ILS
        abc_ils = ABCTSPILS(
            distance_matrix=distance_matrix,
            num_employed_bees=25,
            num_onlooker_bees=50,
            limit=100,
            patience=300,
            local_search_interval=50,
            heuristic_init_ratio=0.8,
            use_parallel=True,
            num_workers=16
        )

        best_tour, best_distance = abc_ils.run(max_iterations=2000)

        print("\n--- FINAL RESULT (ABC + ILS hybrid) ---")
        print("Best tour:", best_tour)
        print(f"Best distance: {best_distance:.2f}")
        if optimal is not None:
            print(f"Gap to optimal: {best_distance - optimal:.2f}")

        global_history.append(best_distance)
    print(global_history)


if __name__ == "__main__":
    main()


