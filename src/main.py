#!/usr/bin/env python3
# encoding: utf-8

import math
import csv
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from ptpdb import set_trace
from scipy.special import expit
from random import randint
from random import randrange

from numpy import log as ln
from numpy import dot
from sklearn import svm
from sklearn.preprocessing import scale as normalize
from sklearn.utils import shuffle

from decision_tree import *
from random_forest import *


# ============= run spam train =============
# cross validate
# run d_tree on spam
# run rnd_forest on spam

# run d_tree on census
# run rnd_forest on census

# create small set to test
# create validation set to test on
# get error
# print error

mat_file = sio.loadmat('./spam_dataset/spam_data.mat')
data = mat_file['training_data']
labels = mat_file['training_labels'][0]



data, labels = shuffle(data, labels)
data, labels = data[10], labels[10]

Tree = DecisionTree(list(range(len(data))))
print("Tree(data[2]): {}".format(Tree.predict((data[2]))))

