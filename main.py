def mf_ant1(G, m):  # todo implement maximal flow function as ant algorithm (Ivan)
    flow = 0
    return flow


def mf_ant2(G, m):  # todo implement maximal flow function as ant algorithm (Jakub)
    flow = 0
    return flow


def mf_ant3(G, m):  # todo implement maximal flow function as ant algorithm (Miron)
    flow = 0
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
    G = []    # graph of flow, where G[0] is source node and G[i] = (a, b) means that i and a are connected with flow b
    m = 10    # number of ants
    ant = [mf_ant1(G, m), mf_ant2(G, m), mf_ant3(G, m)]
    classic = mf_classic(G)
    compare(ant[0], ant[1], ant[2], classic)
