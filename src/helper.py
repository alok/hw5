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
import scipy.stats
import matplotlib as plt
import sklearn
import numpy as np

from pudb import set_trace
from sklearn.utils import shuffle


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


# To make unique filenames to save pickles.

def timestamp():
    return str(time.localtime().tm_min) + '_' + str(time.localtime().tm_sec)


def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

def k_fold(data, labels, k = 2/3):
    shuffled_data, shuffled_labels = shuffle(data, labels)

    t = math.floor(len(data) * k)

    left_data     = shuffled_data[:t]
    right_data  = shuffled_data[t:]

    left_labels     = shuffled_labels[:t]
    right_labels  = shuffled_labels[t:]

    validation_set = (right_data, right_labels)
    training_set = (left_data, left_labels)

    return (training_set, validation_set)
