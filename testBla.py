# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:38:12 2018

@author: Laura
"""

from __future__ import division
import csv
import numpy as np
# a python script define utility function i.e. format source data for computation
# test of this script can be found in testUtility.py



x = np.array([[0.5, 0.5], [0.5, 0.5]])

h = np.ones((1,len(x)))

print(h)
a1 = np.sum(x, axis = 0) == h
a2 = np.sum(x, axis = 1) == h.T
if np.all(a1) and np.all(a2):
    print('this works')
