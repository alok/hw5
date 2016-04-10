#!/usr/bin/env python3
# encoding: utf-8

import csv
import scipy
import sklearn
import numpy as np

from sklearn.feature_extraction import DictVectorizer

from pudb import set_trace

census_data_file = open("./census_data/train_data.csv")
census_data = csv.DictReader(census_data_file)

census_data = list(census_data)


# [] ============= CONSTANTS =============

categorical_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

quant_vars = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',]

vars = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']


# [] ============= Fill in Missing Values =============

def mode(key, lst):
    """
    key -> [ {} ]  -> val
    """
    tmp = []
    for data_pt in lst:
        tmp.append(data_pt[key])
    return scipy.stats.mode(tmp)[0][0]


def fillInMissing(attr_list = vars, lst = census_data, fill_function = mode):
    for attr in attr_list:
        fill_in_value = fill_function(attr, lst)
        for data_pt in lst:
            if data_pt[attr] == '?':
                data_pt[attr] = fill_in_value

    return lst


# [] ============= Process Floats =============

# Replace each string of a float with the float

for i in quant_vars:
    for data_pt in census_data:
        data_pt[i] = float(data_pt[i])


# [] ============= One Hot Encoding =============
# 'one hot encoding' sounds oddly sensual for programming

v = DictVectorizer()
vectorized_dict = v.fit_transform(census_data)

# [] ============= Different Fill In Methods =============
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
