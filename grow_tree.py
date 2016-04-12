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


def all_equal(iterator):
    # guard against empty lists
    if not iterator:
        return False
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(first == rest for rest in iterator)
    except StopIteration:
        return True


def train(I=range(len(data)), y=labels):
    Y = [y[i] for i in I]

    if all_equal(Y):
        return Leaf(label=Y[0])
    else:
        split_rule = segmenter(I)
        split_feature, threshold = split_rule

        I_l = [i for i in I if data[i][split_feature] < threshold]
        I_r = [i for i in I if data[i][split_feature] >= threshold]

        return Node(split_rule, train(I_l), train(I_r))

        self._left_child = Node(left, split_rule=segmenter(left))
        self._right_child = Node(right, split_rule=segmenter(right))
