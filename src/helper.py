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

from pudb import set_trace


def mode(lst):
    """
    returns first mode of list in case of ties
    """
    return scipy.stats.mode(lst)[0][0]


def entropy(I, data, labels):
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
