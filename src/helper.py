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

from inspect import isclass

from pudb import set_trace


def mode(lst):
    """
    returns first mode of list in case of ties
    """
    return scipy.stats.mode(lst)[0][0]
