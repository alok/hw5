#!/usr/bin/env python3
# encoding: utf-8

import math
import random
import csv
import os
import sys
import subprocess
import functools
import itertools

import scipy
import matplotlib as plt
import sklearn
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer

from ptpdb import set_trace

# read in data
# look at it
census_data_file = open("./census_data/train_data.csv")
census_data = csv.DictReader(census_data_file)
print("census_data: {}".format(census_data))
vectorized_dict = DictVectorizer(census_data)
print("vectorized_dict: {}".format(vectorized_dict))


categorical_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# TODO put data into lists and put
# X ->  [
#           [  ]
#                    ]
# (n,m) where n is number of samples and m is number of features
# TODO map th

# Y -> [ {0,1} ] (n,1) or (n,)
# TODO check if quant data is in string format and cast to float if needed
