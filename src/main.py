from copy import deepcopy
from random import randint
from src.graph import Graph, Edge, graph_from_raw_adjacency_list
from random import random

def mf_ant1(G, m):  # todo implement maximal flow function as ant algorithm (Ivan)
    ro = 0.8
    Q = 10
    N = len(G)
    flow = 0
    ant_arcs = []
    fixed_arcs = []
    for j in range(len(G[0])):
        ant_arcs.append((0, G[0][j][0], G[0][j][1]))
    for i in range(1, N-1):
        maxD = 0
        chosen_arc = 0
        for j in range(len(G[i])):
            if G[i][j][1] > maxD:
                maxD = G[i][j][1]
                chosen_arc = j
        fixed_arcs.append((i, G[i][chosen_arc][0], G[i][chosen_arc][1]))
        for j in range(len(G[i])):
            if j != chosen_arc:
                ant_arcs.append((i, G[i][j][0], G[i][j][1]))
    print(ant_arcs)
    print(fixed_arcs)
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
                # print(fl_out[fixed_arcs[i][0]] - fl_in[fixed_arcs[i][0]])
                if fl_in[fixed_arcs[i][0]] - fl_out[fixed_arcs[i][0]] < 0 or fl_in[fixed_arcs[i][0]] - fl_out[fixed_arcs[i][0]] > fixed_arcs[i][2]:
                    restart = True
                    break
                fl = fl_in[fixed_arcs[i][0]] - fl_out[fixed_arcs[i][0]]
                fl_in[fixed_arcs[i][1]] += fl
                fl_out[fixed_arcs[i][0]] += fl
                ant_fl.append(fl)

            if restart or fl_out[0] == 0:
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


def mf_ant2(G, m):  # todo implement maximal flow function as ant algorithm (Jakub)
    flow = 0
    return flow


def mf_ant3(g: Graph, m):  # todo implement maximal flow function as ant algorithm (Miron)
    G = g.adjacency_list_raw()
    tau = [[1.0 for j in range(G[0][i][1]+1)] for i in range(len(G[0]))]
    Q = 10.0
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


def mf_classic(g: Graph | list[list[tuple[int, int]]]):  # todo implement maximal flow minimal cost function in standard way
    def DFS(C, Cmat, s, t):
        def DFS_find(C, v, t, parents, visited):
            if v == t:
                return True
            for u in sorted(C[v], key=lambda x: Cmat[v][x]):
                if not visited[u]:
                    visited[u] = True
                    parents[u] = v
                    if DFS_find(C, u, t, parents, visited):
                        return True
            return False

        parents = [-1 for _ in C]
        visited = [False for _ in C]
        visited[s] = True
        if DFS_find(C, s, t, parents, visited):
            # print(parents)
            max_flow = float('inf')
            current = t
            while current != s:
                if Cmat[parents[current]][current] < max_flow:
                    max_flow = Cmat[parents[current]][current]
                current = parents[current]

            current = t
            while current != s:
                Cmat[parents[current]][current] -= max_flow
                if Cmat[parents[current]][current] == 0:
                    C[parents[current]].remove(current)
                if Cmat[current][parents[current]] == 0:
                    C[current].append(parents[current])
                Cmat[current][parents[current]] += max_flow

                # flow_mat[parents[current]][current] += max_flow
                current = parents[current]
            return max_flow
        else:
            # print(C)
            return 0

    # C = [[] for _ in G.adjacency_list]
    # Cmat = [[0 for _ in G.adjacency_list] for _ in G.adjacency_list]

    # for edge in G.edge_list:
    #     C[edge[0]].append(edge[1])
    #     Cmat[edge[0]][edge[1]] = edge[2]
    C = g.adjacency_list_raw(include_capacity=False)
    Cmat = g.adjacency_matrix_raw()


    # print(C)
    # print(Cmat)

    total_flow = 0
    iteration_flow = 1
    while iteration_flow > 0:
        iteration_flow = DFS(C, Cmat, 0, len(C) - 1)
        total_flow += iteration_flow

    return total_flow



def compare(a1_flow, a2_flow, a3_flow, c_flow):
    print(f"Maximal flow from ant 1 algorithm:    {a1_flow}")
    print(f"Maximal flow from ant 2 algorithm:    {a2_flow}")
    print(f"Maximal flow from ant 3 algorithm:    {a3_flow}")
    print(f"Maximal flow from classic algorithm:  {c_flow}")


if __name__ == '__main__':
    G = [[] for _ in range(9)]    # graph of flow, where G[0] is source node and G[i] = (a, b) means that i and a are
                                  # connected with flow b and G[len(G)-1] is the destination
    m = 10                        # number of ants
    G[0].extend([(1, 5), (2, 7), (3, 3)])
    G[1].extend([(5, 2), (6, 5)])
    G[2].extend([(4, 4)])
    G[3].extend([(4, 2), (7, 8)])
    G[4].extend([(6, 11), (7, 10)])
    G[5].extend([(6, 3), (4, 3)])
    G[6].extend([(8, 6)])
    G[7].extend([(8, 4)])                   # example graph filled
    # print(G)
    # print(graph_from_raw_adjacency_list(G))
    F = G

    G = Graph(9, [Edge(0, 1, 5), Edge(0, 2, 7), Edge(0, 3, 3), Edge(1, 5, 2), Edge(1, 6, 5), Edge(2, 4, 4),
                  Edge(3, 4, 3), Edge(3, 7, 8), Edge(4, 6, 11), Edge(4, 7, 10), Edge(5, 6, 3), Edge(5, 4, 3),
                  Edge(6, 8, 6), Edge(7, 8, 4)])

    g1 = Graph(5, [Edge(0, 1, 3), Edge(0, 2, 2), Edge(2, 3, 1), Edge(1, 3, 3), Edge(1, 4, 2), Edge(2, 4, 20)])
    # print(g1)
    # g2 = Graph(3, [Edge(0, 1, 3), Edge(0, 2, 2)])
    # G = g.adjacency_list_raw()
    # print(g.adjacency_list_raw)
    ant = [mf_ant1(F, m), mf_ant2(G, m), mf_ant3(G, m), mf_classic(G)]
    # classic = mf_classic(G)
    compare(ant[0], ant[1], ant[2], ant[3])
    # print(mf_classic(g1))
