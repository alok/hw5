#!/usr/bin/env python3
# encoding: utf-8

import math
import pickle
import csv
import os
import sys
import scipy
import numpy as np
import scipy.io as sio
from ptpdb import set_trace
from scipy.special import expit
from random import randint
from random import randrange

from numpy import log as ln
from numpy import dot
from sklearn.preprocessing import scale as normalize
from sklearn.utils import shuffle


from random_forest import *

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

def k_fold(data = train, k = 2/3):
    t = math.floor(len(data) * k)

    left     = data[:t]
    current  = data[t:]

    validation_set = current
    training_set = left

    return (training_set, validation_set)
