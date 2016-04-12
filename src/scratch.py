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

    def __init__(self, I, split_rule, left_child=None, right_child=None, leaf_value=None):
        self.I = I
        self._split_rule = split_rule
        self._left_child = left_child
        self._right_child = right_child
        self._leaf_value = leaf_value

    split_feature, threshold = split_rule

    I_l = [i for i in I if data[i][split_feature] < threshold]
    I_r = [i for i in I if data[i][split_feature] >= threshold]

    # if one of the splits is empty, the current node is a leaf
    if not I_l or not I_r:
        self._leaf_value = mode(I)
    else:
        self._left_child = Node(I=I_l, split_rule=segmenter(I_l))
        self._right_child = Node(I=I_r, split_rule=segmenter(I_r))


class Leaf(object):

    """Docstring for Leaf. """

    def __init__(self, val):
        """TODO: to be defined1.

        :val: TODO

        """
        self._val = val
