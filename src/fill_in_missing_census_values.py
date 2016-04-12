#!/usr/bin/env python3
# encoding: utf-8

# ============= Imports =============

import math
import csv
import scipy
import sklearn
import numpy as np

from sklearn.feature_extraction import DictVectorizer

from pudb import set_trace

# ============= Read in File =============

census_data_file = open("./census_data/train_data.csv")
census_data = csv.DictReader(census_data_file)

census_data = list(census_data)

# ================ Constants =============

categorical_vars = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex',
                    'native-country']

quant_vars = ['age', 'fnlwgt', 'education-num', 'capital-gain',
              'capital-loss', 'hours-per-week', ]

vars = categorical_vars + quant_vars

# ================ Encode Labels =============

labels = [x.pop('label') for x in census_data]
labels = [float(y) for y in labels]
labels = np.array(labels)


# ================ Fill in Missing Values =============

def mode(key, lst):
    """
    key -> [ {} ]  -> val
    """
    tmp = []
    for data_pt in lst:
        tmp.append(data_pt[key])
    return scipy.stats.mode(tmp)[0][0]


def fill_in_missing(attr_list = vars, lst = census_data, fill_function = mode):
    for attr in attr_list:
        fill_in_value = fill_function(attr, lst)
        for data_pt in lst:
            if data_pt[attr] == '?':
                data_pt[attr] = fill_in_value

    return lst

# ================ Process Floats =============

# Replace each string of a float with the float
def process_floats(lst = census_data):
    for i in quant_vars:
        for data_pt in lst:
            data_pt[i] = float(data_pt[i])
    return lst


# ================ Actually run the fill and float functions =============

census_data = fill_in_missing()
census_data = process_floats()

# ================ One Hot Encoding =============
# 'one hot encoding' sounds oddly sensual for programming

v = DictVectorizer()
data = v.fit_transform(census_data).toarray()

# ================ Different Fill In Methods =============
# TODO

class Metric():

    """ Different metrics for filling in missing values. """

    def quant(x, y):
        """
        metric for quantitative variables, max(x,y) - min (x,y) / (x + y)
        """

        return ( max(x,y) - min(x,y) ) / (x + y)


    def cat(self, x, y):
        """
        discrete metric for categorical variables
        """
        if x == y:
            return 0
        else:
            return 1
