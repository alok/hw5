#!/usr/bin/env python3
# encoding: utf-8

import math
import random
import os
import sys
import subprocess
import functools
import itertools

import scipy
import matplotlib as plt
import sklearn
import numpy as np

from inspect import isclass

from pudb import set_trace


class Node(object):

    def __init__(self, I, split_rule):
        self.I = I
        self._split_rule = split_rule
        self.l = None
        self.r = None
        self._leaf_value = None

    split_feature, threshold = self.split_rule

    I_l = [i for i in I if data[i][split_feature] < threshold]
    I_r = [i for i in I if data[i][split_feature] >= threshold]

    # if one of the splits is empty, the current node is a leaf
    if not I_l or not I_r:
        self._leaf_value = mode(I)
    else:
        self._left_child = Node(I=I_l, split_rule=segmenter(I_l))
        self._right_child = Node(I=I_r, split_rule=segmenter(I_r))
