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

census_tree_pickle  = open('./pickles/census_tree_pickle.pickle', 'rb')
census_train_pickle = open('./pickles/census_train.pickle',       'rb')
census_test_pickle  = open('./pickles/census_test.pickle',    'rb')

census_tree         = pickle.load(census_tree_pickle)

census_train_data   = pickle.load(census_train_pickle)
census_train_labels = pickle.load(census_train_pickle)


census_test_data   = pickle.load(census_test_pickle)

census_tree_pickle.close()
census_train_pickle.close()
census_test_pickle.close()

train_predictions = []
for pt in census_train_data:
    train_predictions.append(census_tree.predict(pt))
train_predictions = [int(i) for i in train_predictions]

print("benchmark error rate: {:10.2f}%".format(100 * benchmark(train_predictions, census_train_labels)[0]))

test_predictions = []
for pt in census_test_data:
    test_predictions.append(census_tree.predict(pt))
test_predictions = [int(i) for i in test_predictions]

if sys.argv[1] == 'csv':
    kaggle_census = [['id', 'Category']] + [i+1, digit for i, digit, in enumerate(test_predictions)]
    with open('kaggle_census_test_single_tree.csv','w') as fp:
        a = csv.writer(fp, delimiter = ',')
        a.writerows(kaggle_census)
