from time import time
from typing import Callable

from src.ford_fulkerson_algorithm import mf_classic
from src.graph_builder import create_dag
from src.kuba_algorithm import mf_ant3
from src.miron_algorithm import mf_ant2


def run_trials(generated_graphs: int,
               trials_per_graph: int,
               algorithm_iterations: int,
               graph_size: int,
               graph_density: float,
               graph_capacity_range: tuple[int, int]) -> tuple[list[float], list[float]]:

    algorithms: tuple[Callable, Callable, Callable]
    time_tab: list[list[float]]
    ans_tab: list[list[float]]

    algorithms = (mf_ant2, mf_ant3, mf_classic)
    time_tab = [[], [], []]
    ans_tab = [[], [], []]

    for _ in range(generated_graphs):
        G = create_dag(graph_size, density=graph_density, capacity_range=graph_capacity_range)

        for i, algorithm in enumerate(algorithms):
            start = time()
            ans = 0
            for _ in range(trials_per_graph):
                ans += algorithm(G, algorithm_iterations)
            stop = time()
            time_tab[i].append(1000 * (stop - start) / trials_per_graph)
            ans_tab[i].append(ans / trials_per_graph)

    time_tab_avg = [sum(measurements) / len(measurements) for measurements in time_tab]
    ans_tab_avg = [sum(answers) / len(answers) for answers in ans_tab]

    return time_tab_avg, ans_tab_avg


def run_accuracy_trials(trials: int,
                        iterations: int,
                        graph_size: int,
                        graph_density: float,
                        graph_capacity_range: tuple[int, int]) -> tuple[list[float], int]:

    graph = create_dag(graph_size, graph_density, graph_capacity_range)
    histories = [(mf_ant3(graph, iterations, show_history=True)) for _ in range(trials)]

    history_avg = [sum(histories[t][i] for t in range(trials)) / trials for i in range(iterations)]

    return history_avg, mf_classic(graph)
