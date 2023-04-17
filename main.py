def mfmc_ant(G):  # todo implement maximal flow minimal cost function as ant algorithm
    flow = 0
    cost = 0
    return flow, cost


def mfmc_classic(G):  # todo implement maximal flow minimal cost function in standard way
    flow = 0
    cost = 0
    return flow, cost


def compare(a_flow, a_cost, c_flow, c_cost):
    print(f"Maximal flow from ant algorithm:      {a_flow}")
    print(f"Maximal flow from classic algorithm:  {c_flow}")
    print(f"Minimal cost from ant algorithm:      {a_cost}")
    print(f"Minimal cost from classic algorithm:  {c_cost}")


if __name__ == '__main__':
    G = []
    ant = mfmc_ant(G)
    classic = mfmc_classic(G)
    compare(ant[0], ant[1], classic[0], classic[1])
