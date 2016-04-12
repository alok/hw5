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

from fill_in_missing_census_values import data
from fill_in_missing_census_values import labels


def mode(lst):
    """
    returns first mode of list in case of ties
    """
    return scipy.stats.mode(lst)[0][0]


def entropy(I=list(range(len(data)))):
    """ Entropy of an index set. """

    # alias for readability
    y = labels

    p = [len([i for i in I if y[i] == val]) / len(I) for val in set(y)]

    h = p

    for i, pr in enumerate(p):
        if pr == 0:
            # guard against math domain error
            h[i] = 0
        else:
            h[i] = pr * math.log(pr, 2)

    return -np.sum(h)

print("entropy(): {}".format(entropy()))
