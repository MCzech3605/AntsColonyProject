from __future__ import annotations

import random

from src.graph import Graph, Edge


def create_dag(size: int, density: float = 0.2, capacity_range: tuple = (1, 10)) -> Graph:
    graph: Graph = Graph(size, [])
    for start in range(size - 1):
        mandatory_edge_forward_end = random.randint(start + 1, size - 1)
        graph.add_edge(Edge(start, mandatory_edge_forward_end, random.randint(*capacity_range)))
        for end in range(start + 1, size):
            if end == mandatory_edge_forward_end:
                continue
            if random.random() < density:
                graph.add_edge(Edge(start, end, random.randint(*capacity_range)))

    return graph
