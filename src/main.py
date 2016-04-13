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

census_train_pickle = open('./pickles/census_train.pickle', 'rb')

census_train_data = pickle.load(census_train_pickle)
census_train_labels = pickle.load(census_train_pickle)

census_train_pickle.close()

census_train_data, census_train_labels = shuffle( census_train_data, census_train_labels)

# census_train_data, census_train_labels = census_train_data[:10000], census_train_labels[:10000]


Tree = DecisionTree(I=list(range(len(census_train_data))), data=census_train_data, labels=census_train_labels)

census_tree_pickle = open('./pickles/census_tree_pickle.pickle', 'wb')
pickle.dump(Tree, census_tree_pickle)
census_train_pickle.close()
