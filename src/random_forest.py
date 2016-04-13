#!/usr/bin/env python3
# encoding: utf-8

import math
import random
import os
import sys
import subprocess
import functools
import itertools

from math import floor
from collections import Counter
from pudb import set_trace

import scipy
import sklearn
import numpy as np

from helper import *
from decision_tree import *


class RandomForest(object):
    """ takes in an index set I so using random forest is easy"""

    def __init__(self, I, num_trees=5, data, labels):

        self.n = len(data)
        self.num_trees = num_trees
        self.sample_indices_list = [sorted(np.random.choice( \
                                    self.n, floor(2 / 3 * self.n))) \
                                    for _ in range(self.num_trees)]
        self.tree_list = [DecisionTree(I) for I in self.sample_indices_list]

    def predict(self, pt):
        return mode([d_tree.predict(pt) for d_tree in self.tree_list])
