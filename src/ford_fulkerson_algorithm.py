import sys
from collections import deque

import numpy as np

from src.graph import Graph


def mf_classic(g: Graph, iterations: int = 0, v: bool = False) -> int:
    def DFS_find(s: int, t: int, parents: list[int], visited: list[bool]) -> bool:
        queue: deque = deque()
        queue.append(s)
        while queue:
            v = queue.popleft()
            for u, cap in enumerate(Cmat[v]):
                if not visited[u] and cap > 0:
                    visited[u] = True
                    parents[u] = v
                    queue.append(u)

        return visited[t]

    def DFS(C: list[list[int]], Cmat: list[list[int]], s: int, t: int) -> int:

        parents = [-1 for _ in C]
        visited = [False for _ in C]
        visited[s] = True
        if not DFS_find(s, t, parents, visited):
            return 0

        max_flow = sys.maxsize
        current = t
        while current != s:
            if Cmat[parents[current]][current] < max_flow:
                max_flow = Cmat[parents[current]][current]
            current = parents[current]

        current = t
        while current != s:
            Cmat[parents[current]][current] -= max_flow
            Cmat[current][parents[current]] += max_flow

            current = parents[current]

        return max_flow

    C = g.adjacency_list_raw_no_capacity()
    Cmat = g.adjacency_matrix_raw()

    total_flow = 0
    iteration_flow = 1
    while iteration_flow > 0:
        iteration_flow = DFS(C, Cmat, 0, len(C) - 1)
        total_flow += iteration_flow
    if v:
        print(np.array(Cmat))
    return total_flow