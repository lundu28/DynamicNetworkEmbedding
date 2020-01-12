# -*- encoding:utf8 -*-

from random import randrange,random
import numpy as np
from datetime import datetime

class AliasTable():
    def __init__(self, _probs):
        probs = np.array(_probs, dtype = np.float64)
        self.num = len(_probs)
        probs = probs / np.sum(probs)
        #print probs
        self.probs_table = np.ones(self.num, dtype = np.float64)
        self.bi_table = np.zeros(self.num, dtype = np.int32)
        p = 1.0 / self.num
        L, H= [], []
        for i in xrange(self.num):
            if probs[i] < p:
                L.append(i)
            else:
                H.append(i)

        while len(L) > 0 and len(H) > 0:
            l = L.pop()
            h = H.pop()
            self.probs_table[l] = probs[l] / p
            self.bi_table[l] = h
            probs[h] = probs[h] - (p - probs[l])
            if probs[h] < p:
                L.append(h)
            else:
                H.append(h)
        del L, H
        #print self.probs_table
        #print self.bi_table

    def sample(self):
        idx  = randrange(self.num)
        if random() < self.probs_table[idx]:
            return idx
        else:
            return self.bi_table[idx]

if __name__=='__main__':
    test=[1, 1, 3, 2, 4]
    at = AliasTable(test)
    mapp = {}
    for i in xrange(10000):
        t = at.sample()
        if t in mapp:
            mapp[t] += 1
        else:
            mapp[t] = 1
    print mapp
