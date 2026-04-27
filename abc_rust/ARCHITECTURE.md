# Архитектура `abc_rust`

Краткое описание порта гибрида **ABC + ILS** для симметричной **TSP** из Python (`ABCTSPILS` в `ABC.py`) и связанных частей.

## Модули

| Модуль | Роль |
|--------|------|
| `abc.rs` | **`AbcTspIls`**: основной цикл, фазы employed / onlooker / scout, edge memory, kick, diversity, периодический глобальный LS по лучшему туру, учёт времени (`last_run_elapsed`, …). |
| `abc_classic.rs` | **`AbcClassic`**: классический ABC без ILS-слоя (порт `ABC_Classic/ABC_classic.py`). |
| `adaptive.rs` | **`adaptive_parameters(n)`** — те же пороги по `n`, что в `main.py` (`get_adaptive_parameters`), включая ветку 500+. |
| `task_sets.rs` | **`MENU_TASKS`** (как в `main.py`) и **`BENCHMARK_TASKS`** (как в `benchmark.py`; отличаются оптимумы у rd100/rd400). |
| `tsp.rs` | Матрица расстояний, длина тура, фитнес `1/(d+ε)`. |
| `init.rs` | Nearest neighbor, greedy. |
| `perturb.rs` | Double-bridge, multi-insert, случайные 2-opt, `two_opt_swap`. |
| `local_search.rs` | Списки ближайших соседей, `fast_2opt`, `fast_2opt_neighbors`, **`local_search_3opt`** (упрощённый, как в Python `tsp_optimizations.local_search_3opt`). |
| `tsplib.rs` | Чтение **EUC_2D** + `NODE_COORD_SECTION`. |
| `lkh.rs` | Обёртка запуска бинарника LKH (как `LKH/run_lkh.py`). |

## Бинарники

- **`abc-tsp`** — CLI с флагами (`--tsp`, итерации, seed, `--classic`, `--sequential`, `--workers`).
- **`abc-main`** — интерактивное меню выбора задачи, как **`main.py`** (таблица задач → ввод номера → адаптивные параметры → запуск).
- **`abc-benchmark`** — пакетный прогон и CSV, как **`benchmark.py`**.
- **`lkh-run`** — один вызов LKH по `.tsp`.

## Соответствие Python `ABCTSPILS`

**Совпадает по структуре:** те же фазы, инициализация с эвристиками и edge memory, perturb/kick, периодический 2-opt по лучшему туру, адаптация `limit`/`patience`/интервалов.

**Отличия:**

- **Нет GPU** (в Python был опциональный CuPy).
- **Параллельная фаза employed (rayon):** снимок популяции на начало итерации. В Python `ThreadPoolExecutor` + GIL — поведение не идентично построчно.
- **Последовательная фаза employed:** партнёры берутся из уже обновлённых пчёл в текущей итерации (как в последовательной ветке Python).
- **Углублённый Lin–Kernighan** из Python в основной цикл `run()` фактически не подключён; в Rust не портировался.
- **Визуализация** (`matplotlib`) отсутствует.

### Локальный поиск и 3-opt

В **`_adaptive_local_search`** (Python) при **малом `n` (< 200)** и **слабом среднем улучшении** последних 2-opt шагов включается ветка **`local_search_3opt`**. В первом порте Rust она была временно убрана, чтобы быстрее стабилизировать порт (в Python сам 3-opt — упрощённая эвристика с одним шаблоном перестроения и последующим 2-opt swap, не полный 3-opt). **Сейчас эта ветка восстановлена** и вызывает `local_search_3opt`, логика включения та же, что в `ABC.py`.

## Время выполнения

После `initialize_population` фиксируется момент старта; по завершении `run()` заполняется **`last_run_elapsed`** (аналог интервала между `start_time` и концом в Python). В лог печатается строка `Finished. … time = … s`.

## Как проверить LKH (`lkh-run`)

Нужен **собранный** бинарник: в репозитории `LKH/LKH-2.0.7/LKH` (из каталога `LKH/LKH-2.0.7/SRC` выполнить `make`, при необходимости `chmod +x`).

Из каталога `abc_rust/` или корня репозитория:

```bash
cargo build --release

# Только пути: есть ли LKH и (опционально) файл задачи
./target/release/lkh-run --print-paths
./target/release/lkh-run --print-paths --example

# Быстрый тест: eil51, оптимум 426
./target/release/lkh-run --example

# Своя задача
./target/release/lkh-run --tsp ../matrix_task/a280.tsp

# Если не парсится Cost — смотреть сырой вывод солвера
./target/release/lkh-run --example --verbose
```

По умолчанию ищется бинарник LKH, поднимаясь от текущей папки к корню репо; свой путь: `--lkh-bin`.

## Зависимости

`rand`, `rand_chacha`, `clap`, `rayon`, `csv` (для бенчмарка).
