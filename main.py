from copy import deepcopy
from random import randint


def mf_ant1(G, m):  # todo implement maximal flow function as ant algorithm (Ivan)
    flow = 0
    return flow


def mf_ant2(G, m):  # todo implement maximal flow function as ant algorithm (Jakub)
    flow = 0
    return flow


def mf_ant3(G, m):  # todo implement maximal flow function as ant algorithm (Miron)
    tau = [[1.0 for j in range(G[0][i][1]+1)] for i in range(len(G[0]))]
    Q = 10.0
    ro = 0.8

    def calc_max_flow(chosen_vals:list, G:list):
        residual_network = deepcopy(G)
        for i in range(len(residual_network[0])):
            residual_network[0][i] = (residual_network[0][i][0], chosen_vals[i])
        flow = mf_classic(residual_network)
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
                prob_sum += tau_table[ind]
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


def mf_classic(G):  # todo implement maximal flow minimal cost function in standard way
    flow = 0
    return flow


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
    ant = [mf_ant1(G, m), mf_ant2(G, m), mf_ant3(G, m)]
    classic = mf_classic(G)
    compare(ant[0], ant[1], ant[2], classic)
