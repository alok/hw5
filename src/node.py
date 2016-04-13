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
from math import sqrt
import scipy
import sklearn
import numpy as np

from helper import *

from pudb import set_trace


class Node(object):

    def __init__(self, I, split_rule, data, labels, depth, max_depth, feature_bagging, subspace_bagging):
        self.I = I
        self.split_rule = split_rule
        self.l = None
        self.r = None
        self.depth = depth
        self.max_depth = max_depth
        self.feature_bagging = feature_bagging
        self.subspace_bagging = subspace_bagging
        self.leaf_value = None

        split_feature, threshold = self.split_rule
        if self.depth >= max_depth:
            self.leaf_value = self.leaf_value = mode(list([labels[i] for i in I]))

        else:

            I_l = [i for i in I if data[i][split_feature] <= threshold]
            I_r = [i for i in I if data[i][split_feature] > threshold]

            if not I_l or not I_r:
                # if one of the splits is empty, the current node is a leaf
                self.leaf_value = mode(list([labels[i] for i in I]))
            else:
                self.l = Node(I=I_l, split_rule=segmenter(I_l, data, labels), data = data, labels =labels, depth = depth+1, max_depth = max_depth)
                self.r = Node(I=I_r, split_rule=segmenter(I_r, data, labels), data = data, labels =labels, depth = depth+1, max_depth = max_depth)


    def predict(self, pt):
        if self.leaf_value is not None:
            return self.leaf_value
        else:
            feature, threshold = self.split_rule
            if pt[feature] <= threshold:
                return self.l.predict(pt)
            else:
                return self.r.predict(pt)


    def segmenter(self, I, data, labels, feature_bagging, subspace_bagging):
        # find best feat to split on and threshold to minimize weighted entropy
        d = len(data[0])
        n = len(data)
        if feature_bagging:
            d = np.random.choice(d, floor(sqrt(d)))  # sample sqrt of features
        if subspace_bagging:
            n = np.random.choice(d, floor((d/3)))  # sample sqrt of features

        info_gain = float("-inf")
        best_split = None

        for f in range(d):
            possible_thresholds = sorted(set([data[i][f] for i in range(n)]))

        # fnlwgt takes ages
            if len(possible_thresholds)  > 1000:
                continue

            for threshold in possible_thresholds:
                I_l = [i for i in I if data[i][f] <= threshold]
                I_r = [i for i in I if data[i][f] > threshold]

                if not I_l or not I_r:
                    # info gain is 0, so skip
                    candidate_info_gain = 0
                else:
                    candidate_info_gain = purity(I_l, I_r, data, labels)

                if candidate_info_gain > info_gain:
                    info_gain = candidate_info_gain
                    best_split = (f, threshold)

        assert best_split is not None
        print("best_split: {}".format(best_split))
        return best_split

def purity(I_l, I_r, data, labels):

    if not I_l or not I_l:
        return 0

    H = entropy
    parent_entropy = H(I=I_l + I_r, data = data, labels = labels)
    weighted_entropy_of_split = ( len(I_l) * H(I_l, data, labels) + len(I_r) * H(I_r, data, labels)) \
        / (len(I_l) + len(I_r))
    return parent_entropy - weighted_entropy_of_split
