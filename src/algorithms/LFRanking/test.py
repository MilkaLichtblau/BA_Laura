# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:08:41 2018

@author: Laura
"""
import csv
import os
import numpy as np
"""
Creates multiple CSV files each one storing a query from a raw CSV with 
multiple queries inside. 
"""
"""
paths = []

for dirpath, dirnames, files in os.walk('../../scoredDataSets/'):
    if dirnames != []:
        paths += dirnames
        print(paths)

for folder in paths:
    for dirpath, dirname, files in os.walk('../../scoredDataSets/' + folder + '/'):
        print(folder)
        print (files)

"""

scores = [1,2,3,4,5,6]
ranks = [5,6,7,8,9,10]    

u = np.array(scores).reshape(len(scores),1)
v = np.array(ranks).reshape(1,len(ranks))
    
    
vu = v*u

vu = vu.reshape(1,len(u)**2)

    
print (u)
print(v)
print(vu)
print(pMatrix)