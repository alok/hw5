#!/usr/bin/env python3
# encoding: utf-8

import math
import random
import os
import csv
import sys
import subprocess
import functools
import itertools

import scipy
import matplotlib as plt
import sklearn
import numpy as np
from scipy.special import expit

def s(x):
    return np.maximum(1e-8, expit(x))
