from copy import deepcopy
from random import randint
from src.graph import Graph, graph_from_raw_adjacency_list
from random import random
from sortedcollections import OrderedSet
from collections import deque
from time import time
import matplotlib.pyplot as plt
from kuba_algorithm import mf_ant3
from graph_builder import create_dag
import numpy as np


def mf_ant1(G, m = 10):  # implementation of maximal flow function as ant algorithm (Ivan)
    G = G.adjacency_list_raw()
    ro = 0.8
    Q = 10
    N = len(G)

    flow = 0
    maxans = 41.0
    ant_arcs = []
    fixed_arcs = []
    #print (G)
    def min_ost(G):
        N = len(G) - 1
        G2 = [[] for _ in range(N)]
        for i in range(1, N + 1):
            for j in range(len(G[i])):
                u = i
                v = G[i][j][0]
                c = G[i][j][1]
                G2[u-1].append((v-1, c, 0))
                G2[v-1].append((u-1, c, 1))
        start = 0

        used = [0] * N
        cost = [0] * N
        ans = []
        d_set = OrderedSet()
        d_set.add((0, 0, 0, 0))
        #print(d_set)
        while len(d_set) != 0:
            it = d_set[0]
            d_set.discard(it)
            if used[it[1]] == 1:
                continue
            v = it[1]
            if v != 0:
                #print(v, it[2], N)
                #print(G2[v])
                if it[4] == 1:
                    ans.append((v + 1, it[2] + 1, it[3]))
                else:
                    ans.append((it[2] + 1, v + 1, it[3]))
            used[it[1]] = 1
            cost[it[1]] = -it[0]
            for elem in G2[v]:
                to, c = elem[0], elem[1]
                if used[to] == 0:
                    d_set.add((-(cost[v] + c), to, v, c, elem[2]))
        return ans

    #print(min_ost(G))

    used = [0] * N

    def ordering(ost, N):
        ans = []
        ans_side = []
        d_in = [0] * N
        d_out = [0] * N
        used = [0] * len(ost)
        M = len(ost)
        #cnt = 0
        while len(ans) != M:
            for i in range(M):
                if used[i] == 0:
                    #print(ost[i])
                    d_in[ost[i][1]] += 1
                    d_out[ost[i][0]] += 1

            for i in range(M):
                if used[i] == 0:
                    if d_in[ost[i][1]] ==1 and d_out[ost[i][1]] ==0:
                        ans.append((ost[i][0], ost[i][1], ost[i][2]))
                        ans_side.append(1)
                        used[i] = 1
                    elif d_out[ost[i][0]] ==1 and d_in[ost[i][0]] ==0:
                        ans.append((ost[i][0], ost[i][1], ost[i][2]))
                        ans_side.append(0)
                        used[i] = 1
            a = 3
            for i in range(N):
                d_in[i] = 0
                d_out[i] = 0

        return (ans, ans_side)

    ost_arcs = min_ost(G)
    #print(ost_arcs)
    #print(ordering(ost_arcs, N))
    fixed_arcs, arc_side = ordering(ost_arcs, N)



    for j in range(len(G[0])):
        ant_arcs.append((0, G[0][j][0], G[0][j][1]))
    for i in range(1, N-1):
        # maxD = 0
        # chosen_arc = 0
        # for j in range(len(G[i])):
        #     if G[i][j][1] > maxD:
        #         maxD = G[i][j][1]
        #         chosen_arc = j
        # fixed_arcs.append((i, G[i][chosen_arc][0], G[i][chosen_arc][1]))
        for j in range(len(G[i])):
            if (i, G[i][j][0], G[i][j][1]) not in fixed_arcs:
                ant_arcs.append((i, G[i][j][0], G[i][j][1]))
    #print(ant_arcs)
    # print(fixed_arcs)
    fixed_arcs = ost_arcs
    pher = [[1.0 for j in range(ant_arcs[i][2] + 1)] for i in range(len(ant_arcs))]

    def get_ant_fl(pher, i):
        prob = random()
        sum_prob = 0.0
        sum_pher = 0.0
        for j in range(len(pher[i])):
            sum_pher += pher[i][j]
        for j in range(len(pher[i])):
            pher_prob = pher[i][j] / sum_pher

            if sum_prob <= prob and prob <= sum_prob + pher_prob:
                return j
            sum_prob += pher_prob

    #def satisfy(ant_fl, fl_in, ):

    for _ in range(m):

        while True:
            fl_in = [0.0 for i in range(N)]
            fl_out = [0.0 for i in range(N)]
            ant_fl = []

            for i in range(len(ant_arcs)):
                fl = get_ant_fl(pher, i)
                fl_in[ant_arcs[i][1]] += fl
                fl_out[ant_arcs[i][0]] += fl
                ant_fl.append(fl)
            # print(fl_in)
            # print(fl_out)
            restart = False
            for i in range(len(fixed_arcs)):
                # print(arc_side[i])
                # print(fl_out[fixed_arcs[i][0]] - fl_in[fixed_arcs[i][0]])
                # print(fl_out[fixed_arcs[i][1]] - fl_in[fixed_arcs[i][1]])
                fl = 0
                if arc_side[i] == 1:
                    if fl_in[fixed_arcs[i][0]] - fl_out[fixed_arcs[i][0]] < 0 or fl_in[fixed_arcs[i][0]] - fl_out[fixed_arcs[i][0]] > fixed_arcs[i][2]:
                        restart = True
                        break
                    fl = fl_in[fixed_arcs[i][0]] - fl_out[fixed_arcs[i][0]]
                else:
                    if fl_out[fixed_arcs[i][1]] - fl_in[fixed_arcs[i][1]] < 0 or fl_out[fixed_arcs[i][1]] - fl_in[fixed_arcs[i][1]] > fixed_arcs[i][2]:
                        restart = True
                        break
                    fl = fl_out[fixed_arcs[i][1]] - fl_in[fixed_arcs[i][1]]
                fl_in[fixed_arcs[i][1]] += fl
                fl_out[fixed_arcs[i][0]] += fl
                ant_fl.append(fl)

            if restart or fl_out[0] == 0:
                for i in range(len(pher)):
                    for j in range(len(pher[i])):
                        pher[i][j] *= ro
                continue

            # print("Res:")
            # print(fl_out)
            # print(fl_in)
            # print(ant_fl)

            result = fl_out[0]
            flow = max(flow, result)
            for i in range(len(pher)):
                for j in range(len(pher[i])):
                    pher[i][j] *= ro
                    if ant_fl[i] == j:
                        pher[i][j] += (result / Q)
            #print(pher)
            break
    return flow
    #return min(flow, maxans)


def mf_ant2(g: Graph, m: int = 10):  # implementation of maximal flow function as ant algorithm (Miron)
    G = g.adjacency_list_raw()
    tau = [[1.0 for j in range(G[0][i][1]+1)] for i in range(len(G[0]))]
    Q = 30.0
    ro = 0.8

    def calc_max_flow(chosen_vals: list, G: list):
        residual_network = deepcopy(G)

        for i in range(len(residual_network[0])):
            residual_network[0][i] = (residual_network[0][i][0], chosen_vals[i])
        flow = mf_classic(graph_from_raw_adjacency_list(residual_network))
        if flow < sum(chosen_vals):
            return 0
        return flow

    def iterate(G, tau_table):
        const = 100000
        chosen_vals = []
        for i in range(len(tau_table)):
            prob_sum = sum(tau_table[i])
            prob_sum *= const
            prob_sum = int(prob_sum)
            rand = randint(0, prob_sum)
            rand = float(rand)/const
            prob_sum = tau_table[i][0]
            ind = 0
            while prob_sum < rand and ind < len(tau_table):
                ind += 1
                prob_sum += tau_table[i][ind]
            chosen_vals.append(ind)
        flow = calc_max_flow(chosen_vals, G)
        for i in range(len(tau_table)):
            tmp = tau_table[i][chosen_vals[i]]
            tau_table[i][chosen_vals[i]] = tmp*ro + flow/Q
        return flow
    flow = 0
    for i in range(m):
        flow = max(flow, iterate(G, tau))
    return flow


def mf_classic(g: Graph | list[list[tuple[int, int]]], *args, v: bool = False):
    def DFS_find(s, t, parents, visited):
        queue = deque()
        queue.append(s)
        # stack = [s]
        while queue:
            v = queue.popleft()
            # for u in sorted(C[v], key=lambda x: Cmat[v][x]):
            for u, cap in enumerate(Cmat[v]):
                if not visited[u] and cap > 0:
                    visited[u] = True
                    parents[u] = v
                    queue.append(u)

        return visited[t]

    def DFS(C, Cmat, s, t):

        parents = [-1 for _ in C]
        visited = [False for _ in C]
        visited[s] = True
        if not DFS_find(s, t, parents, visited):
            return 0

        max_flow = float('inf')
        current = t
        while current != s:
            if Cmat[parents[current]][current] < max_flow:
                max_flow = Cmat[parents[current]][current]
            current = parents[current]

        current = t
        while current != s:
            Cmat[parents[current]][current] -= max_flow
            # if Cmat[parents[current]][current] == 0:
            #     C[parents[current]].remove(current)
            # if Cmat[current][parents[current]] == 0:
            #     C[current].append(parents[current])
            Cmat[current][parents[current]] += max_flow

            # flow_mat[parents[current]][current] += max_flow
            current = parents[current]

        return max_flow

    C = g.adjacency_list_raw(include_capacity=False)
    Cmat = g.adjacency_matrix_raw()

    total_flow = 0
    iteration_flow = 1
    while iteration_flow > 0:
        iteration_flow = DFS(C, Cmat, 0, len(C) - 1)
        total_flow += iteration_flow
    if v:
        print(np.array(Cmat))
    return total_flow


def compare(a1_flow, a2_flow, a3_flow, c_flow):
    print(f"Maximal flow from ant 1 algorithm:    {a1_flow}")
    print(f"Maximal flow from ant 2 algorithm:    {a2_flow}")
    print(f"Maximal flow from ant 3 algorithm:    {a3_flow}")
    print(f"Maximal flow from classic algorithm:  {c_flow}")


def plot_results(time_tab, ans_tab, number_of_algorithms, graph_size):
    fig, ax = plt.subplots()
    ax.grid()
    if number_of_algorithms == 3:
        ax.bar([1, 2, 3], time_tab, tick_label=["Alg2", "Alg3", "Classic"])
    else:
        ax.bar([1, 2, 3, 4], time_tab, tick_label=["Alg1", "Alg2", "Alg3", "Classic"])
    ax.set_title("Average time taken")
    ax.set_ylabel("Time [ms]")
    fig.show()
    fig.savefig(f"Time_lin_{number_of_algorithms}_{graph_size}.png")
    ax.set_yscale('log')
    fig.show()
    fig.savefig(f"Time_log_{number_of_algorithms}_{graph_size}.png")

    fig, ax = plt.subplots()
    ax.grid()
    if number_of_algorithms == 3:
        ax.bar([1, 2, 3], ans_tab, tick_label=["Alg2", "Alg3", "Classic"])
    else:
        ax.bar([1, 2, 3, 4], ans_tab, tick_label=["Alg1", "Alg2", "Alg3", "Classic"])
    ax.set_title("Average flow reached")
    ax.set_ylabel("Flow [units]")
    fig.show()
    fig.savefig(f"Flow_lin_{number_of_algorithms}_{graph_size}.png")


def run_trials(generated_graphs: int,
               trials_per_graph: int,
               algorithm_iterations: int,
               graph_size: int,
               graph_density: float,
               graph_capacity_range: tuple[int, int],
               use_alg1: bool = False) -> tuple[list[list], list[list]]:

    algorithms = (mf_ant2, mf_ant3, mf_classic)
    time_tab = [[], [], []]
    ans_tab = [[], [], []]
    if use_alg1:
        algorithms = (mf_ant1, mf_ant2, mf_ant3, mf_classic)
        time_tab.append([])
        ans_tab.append([])
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

    time_tab = [sum(measurements) / len(measurements) for measurements in time_tab]
    ans_tab = [sum(answers) / len(answers) for answers in ans_tab]

    return time_tab, ans_tab


def plot_accuracy_by_iterations(trials: int,
                                iterations: int,
                                graph_size: int,
                                graph_density: float,
                                graph_capacity_range: tuple[int, int]) -> None:
    graph = create_dag(graph_size, graph_density, graph_capacity_range)
    histories = [(mf_ant3(graph, iterations, show_history=True)) for _ in range(trials)]

    history_avg = [sum(histories[t][i] for t in range(trials)) / trials for i in range(iterations)]
    optimal = mf_classic(graph)
    plt.plot(history_avg, label="Ant")
    plt.plot([optimal for _ in range(iterations)], label="Ford-Fulkerson")
    plt.xlabel("Iterations")
    plt.ylabel("Maximum flow found")
    plt.title("Accuracy of an ant algorithm by iterations")
    plt.legend()
    plt.savefig("convergence.png")
    plt.show()


def main():
    plot_accuracy_by_iterations(20, 200, 50, 0.2, (3, 25))
    exit(0)

    time_tab, ans_tab = run_trials(20, 5, 200, 10, 0.3, (2, 10))
    plot_results(time_tab, ans_tab, 3, 10)

    time_tab, ans_tab = run_trials(20, 5, 200, 50, 0.2, (3, 25))
    plot_results(time_tab, ans_tab, 3, 50)



    # todo dołożyć więcej grafów
    # todo dokument i prezentacja
    # todo znaleźć literaturę


if __name__ == '__main__':
    main()
