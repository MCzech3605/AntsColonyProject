from random import random

from sortedcollections import OrderedSet

from src.graph import Graph


# mypy: ignore-errors

def mf_ant1(graph: Graph, m: int = 10) -> int:  # implementation of maximal flow function as ant algorithm (Ivan)
    G = graph.adjacency_list_raw()
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