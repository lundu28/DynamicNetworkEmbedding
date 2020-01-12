import networkx as nx
import os
import sys

class GetNext(object):
    def __init__(self, params):
        self.f = open(params["input_file"], "r")
        self.is_directed = params["is_directed"]
        self.n = params["num_new_nodes"]

    @staticmethod
    def dict_add(d, key, add):
        if key in d:
            d[key] += add
        else:
            d[key] = add

    def get_next(self, G):
        for num in xrange(self.n):
            while True:
                line = self.f.readline()
                if not line:
                    return num
                line = line.strip()
                if len(line) == 0:
                    continue
                u, m = [int(i) for i in line.split()]
                break
        
            for i in xrange(m):
                line = self.f.readline()
                line = line.strip()
                u, v = [int(i) for i in line.split()]
                G.add_edge(u, v, weight = 1)
                GetNext.dict_add(G.node[u], 'out_degree', 1)
                GetNext.dict_add(G.node[v], 'in_degree', 1)
                GetNext.dict_add(G.graph, 'degree', 1)
                if not self.is_directed and u != v:
                    G.add_edge(v, u, weight = 1)
                    GetNext.dict_add(G.node[v], 'out_degree', 1)
                    GetNext.dict_add(G.node[u], 'in_degree', 1)
                    G.graph['degree'] += 1

        return self.n

    def __del__(self):
        self.f.close()

