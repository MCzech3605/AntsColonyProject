from __future__ import annotations
import numpy as np
from src.graph import Graph

evaporation_rate: float = 0.9
exploitation_factor: float = 0.8
exploration_factor: float = 0.2


def mf_ant3(graph: Graph, iterations: int = 20, ants: int = 15,
            v: bool = False, show_history: bool = False) -> int | list[int]:
    pheromones_this_iteration: np.ndarray
    pheromones_next_iteration: np.ndarray
    adjacency_list: list[np.ndarray]
    base_capacities: np.ndarray

    pheromones_this_iteration = np.ones((graph.size, graph.size))
    adjacency_list = [np.array(l, dtype=int) for l in graph.adjacency_list_raw(include_capacity=False)]

    max_flow = 0
    flows_history = []

    for i in range(iterations):
        flow, pheromones_this_iteration = iteration(graph, adjacency_list, ants, pheromones_this_iteration, v)
        # print(f"iteration {i}: max flow found: {flow}")
        pheromones_this_iteration *= evaporation_rate
        max_flow = max(max_flow, flow)
        flows_history.append(max_flow)

    if show_history:
        return flows_history
    return max_flow


def iteration(graph: Graph, adjacency_list: list[np.ndarray], ants: int,
              pheromones_this_iteration: np.ndarray, v: bool = False) -> (int, np.ndarray):
    pheromones_next_iteration: np.ndarray = np.copy(pheromones_this_iteration)
    capacities_left: np.ndarray = np.array([[edge.capacity if edge is not None else 0.0 for edge in row] for row in graph.adjacency_matrix], dtype=int)

    sink_vertex = graph.size - 1
    flow = 0

    for ant in range(ants):
        if v:
            print(f"\tant {ant} running...")
            print(capacities_left)
        current_vertex = 0
        current_path = []
        current_path_vertices = set()
        current_path_capacity = np.iinfo(np.int32).max

        while current_vertex != sink_vertex:
            options = adjacency_list[current_vertex]
            options_probabilities = (np.power(capacities_left[current_vertex, options], exploration_factor) *
                                     np.power(pheromones_this_iteration[current_vertex, options], exploitation_factor))
            if np.sum(options_probabilities) == 0:
                current_path_capacity = 0
                break

            options_probabilities = options_probabilities / np.sum(options_probabilities)
            next_vertex = np.random.choice(options, p=options_probabilities)

            current_path.append((current_vertex, next_vertex))
            current_path_vertices.add(current_vertex)
            current_path_capacity = min(current_path_capacity, capacities_left[current_vertex, next_vertex])

            current_vertex = next_vertex

            if current_vertex in current_path_vertices:
                raise Exception("graph not acyclic!")

        current_path_indexing = tuple(np.array(current_path).T)
        if current_path_indexing != ():
            current_path_indexing_reverse = (current_path_indexing[1], current_path_indexing[0])
            capacities_left[current_path_indexing_reverse] += current_path_capacity
        capacities_left[current_path_indexing] -= current_path_capacity
        pheromones_next_iteration[current_path_indexing] += current_path_capacity / len(adjacency_list) * 3

        flow += current_path_capacity
        if v:
            print(ant, flow, current_path, current_path_capacity)
    return flow, pheromones_next_iteration
