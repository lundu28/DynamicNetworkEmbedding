import os
import sys
import time
import networkx as nx
import json
import numpy as np
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier


from env import *
from data_handler import DataHandler as dh
from lib_ml import MachineLearningLib as mll

class Metric(object):

    @staticmethod
    def multilabel_classification(X, params):
        X_scaled = scale(X)
        y = getattr(dh, params['load_ground_truth_func'])(os.path.join(DATA_PATH, params["ground_truth"]))
        y = y[:len(X)]
        f1_micro = 0.0
        f1_macro = 0.0

        for _ in xrange(params['times']):
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=params['test_size'])
            log = MultiOutputClassifier(SGDClassifier(loss='log'), n_jobs=params['n_jobs'])
            log.fit(X_train, y_train)

            for i in range(y_test.shape[1]):
                f1_micro += f1_score(y_test[:, i], log.predict(X_test)[:, i], average='micro')
                f1_macro += f1_score(y_test[:, i], log.predict(X_test)[:, i], average='macro')
        return f1_micro/(float(params['times']*y.shape[1])), f1_macro/(float(params['times']*y.shape[1]))
    
    @staticmethod
    def classification(X, params):
        X_scaled = scale(X)
        y = dh.load_ground_truth(os.path.join(DATA_PATH, params["ground_truth"]))
        y = y[:len(X)]
        acc = 0.0
        for _ in xrange(params["times"]):
             X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = params["test_size"], stratify = y)
             clf = getattr(mll, params["classification"]["func"])(X_train, y_train, params["classification"])
             acc += mll.infer(clf, X_test, y_test)[1]
        acc /= float(params["times"])
        return acc

if __name__ == '__main__':
    X = np.random.uniform(-0.1, 0.1, 16).reshape(8, 2)
    drawer = {}
    drawer['func'] = 'abc'
    draw_cnt = 1
    Metric.draw_circle_2D(X, drawer, '', 1)
