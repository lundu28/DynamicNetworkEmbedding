import os
import sys
import re
import json
import math
import argparse
import time
import subprocess
import numpy as np
import networkx as nx
import tensorflow as tf
import datetime
from operator import itemgetter
import random
random.seed(157)
np.random.seed(157)

from utils.env import *
from utils.metric import Metric
from utils.data_handler import DataHandler as dh

FILE_PATH = os.path.dirname(os.path.abspath(__file__))



def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--operation', type = str, default = "all", help = "[all | init | train]")
    parser.add_argument('--conf', type = str, default = "default")
    args = parser.parse_args()
    params = dh.load_json_file(os.path.join(CONF_PATH, args.conf + ".json"))

    metric_path_pre = os.path.join(RES_PATH, args.conf)
    if os.path.exists(metric_path_pre) == False:
        os.mkdir(metric_path_pre)
    output_path = os.path.join(metric_path_pre, dh.get_time_str())
    metric_path = output_path + "_metric"

    def metric(embeddings):
        if "metrics" not in params:
            return
        for metric in params["metrics"]:
            print("[] Start node classification...")
            res = getattr(Metric, metric["func"])(embeddings, metric)
            dh.append_to_file(metric_path, str(res) + "\n")
            print("[+] Metric: " + str(res))
    dh.symlink(metric_path, os.path.join(metric_path_pre, "new_metric"))


    if args.operation == "all":
        G, embeddings, weights = __import__(
                "init." + params["init"]["func"],
                fromlist = ["init"]
                ).init(params["init"], metric, output_path)
        __import__(
                "dynamic_loop." + params["main_loop"]["func"],
                fromlist = ["dynamic_loop"]
                ).loop(params["main_loop"], G, embeddings, weights, metric, output_path)
    elif args.operation == "init":
        G, embeddings, weights = __import__("init." + params["init"]["func"], fromlist = ["init"]).init(params["init"], metric, output_path)
    else:
        print "Not Support!"

if __name__ == "__main__":
    main()
