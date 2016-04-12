#!/usr/bin/env python3
# encoding: utf-8

import math
import random
import os
import sys
import subprocess
import functools
import itertools

from collections import Counter
import scipy
import matplotlib as plt
import sklearn
import numpy as np

from helper import *
from node import Node


from pudb import set_trace


class DecisionTree(object):
    """ takes in an index set I so using random forest is easy"""

    def __init__(self, I):
        self.root = Node(I, split_rule = segmenter(I))


def impurity(I_l, I_r):
    H = entropy

    weighted_entropy_of_split = ( len(I_l) * H(I_l) + len(I_r) * H(I_r)) \
                                    / (len(I_l) + len(I_r))
    return weighted_entropy_of_split


def segmenter(I, feature_bagging = False):
    # find best feat to split on and threshold to minimize weighted entropy
    if feature_bagging:
        n = # sample sqrt of features
    else:
        n = # use all features
    # sort on quant feature when evaluating split
    # how to check if feat is quant? 
