import sys
import os
import json
import numpy as np
import time
import datetime

from utils.env import *
from utils.data_handler import DataHandler as dh

def init(params, metric, output_path):
    # load graph structure
    def load_data(params):
        params["network_file"] = os.path.join(DATA_PATH, params["network_file"])
        G = getattr(dh, params["func"])(params)
        return G

    print("[] Loading data...")
    G = load_data(params["load_data"])
    print("[+] Loaded data!")

    print("[] Initializing embeddings with the original network...")
    module_embedding = __import__(
            "init_embedding." + params["init_train"]["func"], fromlist = ["init_embedding"]).NodeEmbedding
    ne = module_embedding(params["init_train"], G)
    embeddings, weights = ne.train()
    print("[+] Finished initializing!")

    with open(output_path + "_init", "w") as f:
        f.write(json.dumps({"embeddings": embeddings.tolist(), "weights": weights.tolist()}))
    metric(embeddings)
    return G, embeddings, weights

