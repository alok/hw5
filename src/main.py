#!/usr/bin/env python3
# encoding: utf-8

import math
import time
import pickle
import csv
import os
import sys
import scipy
import numpy as np
import scipy.io as sio
from ptpdb import set_trace
from random import randint
from random import randrange

from sklearn.utils import shuffle

from random_forest import *
from helper import *


def arborDay(x):
    
print("================= Hyperparameters ==========================================")

max_depth = 8
num_trees = 10
feature_bagging = False
subspace_bagging = False
depth_bagging = False

if sys.argv[1]:
    max_depth = int(sys.argv[1])
if sys.argv[2]:
    num_trees = int(sys.argv[2])
if sys.argv[3] == 't':
    feature_bagging  = True
if sys.argv[4] == 't':
    subspace_bagging = True
if sys.argv[5] == 't':
    depth_bagging = True

print("max_depth: {}".format(max_depth))
print("num_trees: {}".format(num_trees))
print("feature_bagging: {}".format(feature_bagging))
print("subspace_bagging: {}".format(subspace_bagging))
print("depth_bagging: {}".format(depth_bagging))

print("============= Load Spam ===================================================")

# 'training_labels', 'training_data', 'test_data',

print("============= Load Spam Training Set =============")

spam_mat_file = sio.loadmat('./spam_dataset/spam_data.mat')
spam_train_data = spam_mat_file['training_data']
spam_train_labels = spam_mat_file['training_labels'][0]

print("============= Load Spam Test Set =============")

spam_test_data = spam_mat_file['test_data']

print("============= Load Census ===================================================")


print("============= Load Census Training Set =============")

census_train_pickle = open('./pickles/census_train.pickle', 'rb')

census_train_data = pickle.load(census_train_pickle)
census_train_labels = pickle.load(census_train_pickle)

census_train_pickle.close()

print("============= Load Census Test Set =============")

census_test_pickle  = open('./pickles/census_test.pickle',    'rb')

census_test_data   = pickle.load(census_test_pickle)

census_test_pickle.close()

print("============= Spam Kaggle =============")
# tree
# forest

# print error

print("============= Spam Cross Validate =============")
# tree
# forest

# print error

print("============= Spam Test =============")
# tree
# forest

# print error

print("============= Census  Kaggle =============")
# tree
# forest

# print error

print("============= Census Cross Validate =============")
# tree
# forest

# print error

print("============= Census Test =============")
# tree
# forest

# print error

census_train_data, census_train_labels = shuffle(census_train_data, census_train_labels)
census_train_data, census_train_labels = census_train_data[:100], census_train_labels[:100]

Forest = RandomForest(I = list(range(len(census_train_data))), data = census_train_data, labels = census_train_labels, max_depth = max_depth, num_trees = num_trees, feature_bagging = feature_bagging, subspace_bagging = subspace_bagging, depth_bagging = depth_bagging)
p = census_train_data[2]
print("census_train_labels[2]: {}".format(census_train_labels[2]))
print("Forest.predict(p): {}".format(Forest.predict(p)))

# Tree = DecisionTree(I=list(range(len(census_train_data))), data=census_train_data, labels=census_train_labels, max_depth = max_depth, feature_bagging = feature_bagging, subspace_bagging = subspace_bagging)

