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

    def __init__(self, I, data, labels, max_depth, num_trees, feature_bagging, subspace_bagging, depth_bagging):

        # TODO add random max depth for each tree 'if depth_bagging'
        self.depth_bagging = depth_bagging
        self.n = len(data)
        self.num_trees = num_trees
        if self.depth_bagging:
            self.tree_depths = np.random.choice(30, num_trees)
        else:
            self.tree_depths = [max_depth] * num_trees
        self.sample_indices_list = [sorted(np.random.choice( \
                                    self.n, floor(2 / 3 * self.n))) \
                                    for _ in range(self.num_trees)]

        self.forest = [DecisionTree(I, data, labels, depth, feature_bagging, subspace_bagging) \
                          for depth in self.tree_depths for I in self.sample_indices_list]

    def predict(self, pt):
        return int(mode([d_tree.predict(pt) for d_tree in self.forest]))
