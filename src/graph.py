from __future__ import annotations
from collections import namedtuple
from typing import Optional

Edge = namedtuple('Edge', ['start', 'end', 'capacity'])


def graph_from_raw_adjacency_list(adjacency_list: list[list[tuple[int, int]]]) -> Graph:
    return Graph(len(adjacency_list),
                 [Edge(index, node, capacity) for index, row in enumerate(adjacency_list) for node, capacity in row])


class Graph:
    adjacency_list: list[list[Edge]]
    adjacency_matrix: list[list[Optional[Edge]]]
    edge_list: list[Edge]

    size: int

    def __init__(self, init_size: int, edges: list[Edge]) -> None:
        self.adjacency_list = []
        self.adjacency_matrix = []
        self.size = 0
        for _ in range(init_size):
            self.add_node()

        for edge in edges:
            self.add_edge(edge)

        self.edge_list = edges

    def adjacency_list_raw(self, include_capacity: Optional[bool] = True) -> list[list[tuple[int, int]]]:
        if include_capacity:
            return [[(edge.end, edge.capacity) for edge in node_neighbours] for node_neighbours in self.adjacency_list]
        else:
            return [[edge.end for edge in node_neighbours] for node_neighbours in self.adjacency_list]

    def edge_list_raw(self) -> list[tuple[int, int, int]]:
        return [(edge.start, edge.end, edge.capacity) for edge in self.edge_list]

    def adjacency_matrix_raw(self) -> list[list[int]]:
        return [[edge.capacity if edge is not None else -1 for edge in row] for row in self.adjacency_matrix]

    def add_node(self) -> None:
        self.size += 1
        self.adjacency_list.append([])
        for row in self.adjacency_matrix:
            row.append(None)
        self.adjacency_matrix.append([None for _ in range(self.size)])

    def add_edge(self, edge: Edge) -> None:
        if edge.start >= self.size or edge.start >= self.size:
            raise ValueError("edge out of bounds")
        self.adjacency_list[edge.start].append(edge)
        self.adjacency_matrix[edge.start][edge.end] = edge

    def __repr__(self) -> str:
        row_strings = [f"{i}: {'  '.join([f'<{edge.capacity}> {edge.end}' for edge in connections])}" for i, connections in enumerate(self.adjacency_list)]
        return "\n".join(row_strings) + "\n"
