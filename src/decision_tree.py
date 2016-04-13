#!/usr/bin/env python3
# encoding: utf-8

from collections import Counter
import scipy
import matplotlib as plt
import sklearn
import numpy as np
from pudb import set_trace

from helper import *
from node import *


class DecisionTree(object):
    """ takes in an index set I so using random forest is easy"""

    def __init__(self, I, data, labels):
        self.root = Node(I, split_rule=segmenter(I, data, labels), data = data, labels = labels)

    def predict(self, pt):
        return self.root.predict(pt)
