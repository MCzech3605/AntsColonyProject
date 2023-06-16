from copy import deepcopy
from random import randint

from src.ford_fulkerson_algorithm import mf_classic
from src.graph import graph_from_raw_adjacency_list, Graph


# mypy: ignore-errors

def mf_ant2(g: Graph, m: int = 10) -> int:  # implementation of maximal flow function as ant algorithm (Miron)
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
