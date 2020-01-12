import os
import sys
import networkx as nx
import re
import json
import numpy as np
import math
from datetime import datetime
from Queue import Queue
from sklearn.preprocessing import MultiLabelBinarizer

class DataHandler(object):
    @staticmethod
    def dict_add(d, key, add):
        if key in d:
            d[key] += add
        else:
            d[key] = add

    @staticmethod
    def load_unweighted_digraph(params):
        G = nx.DiGraph()
        with open(params["network_file"], "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                if len(items) != 2:
                    continue
                G.add_edge(int(items[0]), int(items[1]), weight = 1)
                DataHandler.dict_add(G.node[int(items[0])], 'out_degree', 1)
                DataHandler.dict_add(G.node[int(items[1])], 'in_degree', 1)
                DataHandler.dict_add(G.graph, 'degree', 1)
                if not params["is_directed"] and items[0] != items[1]:
                    G.add_edge(int(items[1]), int(items[0]), weight = 1)
                    DataHandler.dict_add(G.node[int(items[1])], 'out_degree', 1)
                    DataHandler.dict_add(G.node[int(items[0])], 'in_degree', 1)
                    G.graph['degree'] += 1
        return G

    @staticmethod
    def out_degree_distribution(G, params = None):
        ret = [0.0] * G.number_of_nodes()
        for u in G:
            ret[u] = G.node[u]['out_degree']
        return ret

    @staticmethod
    def in_degree_distribution(G, params = None):
        ret = [0.0] * G.number_of_nodes()
        for u in G:
            ret[u] = G.node[u]['in_degree']
        return ret

    @staticmethod
    def in_degree_distribution_init(G, num_nodes, params = None):
        ret = [0.0] * num_nodes
        for u in G:
            if u >= num_nodes:
                continue
            ret[u] = G.node[u]['in_degree']
        return ret

    @staticmethod
    def load_graph(file_path):
        G = nx.Graph()
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                if len(items) != 2:
                    continue
                G.add_edge(int(items[0]), int(items[1]))
        return G
    
    @staticmethod
    def load_fea(file_path):
        X = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                items = line.split()
                if len(items) < 1:
                    continue
                X.append([float(item) for item in items])
        return np.array(X)

    @staticmethod
    def transfer_to_matrix(graph):
        n = graph.number_of_nodes()
        mat = np.zeros([n, n])
        for e in graph.edges():
            mat[e[0]][e[1]] = 1
            mat[e[1]][e[0]] = 1
        return mat

    @staticmethod
    def transfer_to_nx(g_mat):
        G = nx.Graph()
        for i in xrange(len(g_mat)):
            for j in xrange(len(g_mat[i])):
                if g_mat[i][j] == 1:
                    G.add_edge(i, j)
        return G

    @staticmethod
    def normalize_adj_matrix(g):
        # diagonal should be 1
        mat_ret = g / np.sum(g, axis = 1, keepdims = True)
        return mat_ret

    @staticmethod
    def symlink(src, dst):
        try:
            os.symlink(src, dst)
        except OSError:
            os.remove(dst)
            os.symlink(src, dst)


    @staticmethod
    def load_json_file(file_path):
        with open(file_path, "r") as f:
            s = f.read()
            s = re.sub('\s',"", s)
        return json.loads(s)

    @staticmethod
    def get_time_str():
        return datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")

    @staticmethod
    def append_to_file(file_path, s):
        with open(file_path, "a") as f:
            f.write(s)

    @staticmethod
    def load_ground_truth(file_path):
        lst = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                lst.append([int(i) for i in items])
        lst.sort()
        return [i[1] for i in lst]
    
    @staticmethod
    def load_multilabel_ground_truth(file_path):
        lst = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                lst.append([int(i) for i in items])
        lst.sort()
        lst = [i[1:] for i in lst]
        mlb = MultiLabelBinarizer()
        return mlb.fit_transform(lst)

    @staticmethod
    def load_onehot_ground_truth(file_path):
        lst = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                lst.append([int(i) for i in items])
        lst.sort()
        return np.array([i[1:] for i in lst], dtype=int)

