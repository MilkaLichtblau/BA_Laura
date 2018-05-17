# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:08:41 2018

@author: Laura
"""
import csv
import numpy as np
import os

print(os.path.exists('../../preprocessedDataSets/GermanCredit_age25pre.csv'))

"""
try:
    with open('GermanCredit_age25.csv', newline='') as File:  
        reader = csv.reader(File)
        rWithHeader = np.array([row for row in reader])
        print('file works')
except FileNotFoundError:
    raise FileNotFoundError("File could not be found. Please enter a valid path to a csv file.")
"""    
    
sorted_score_hat = [8,4,3,4,5,3,9,10,22,1,2,4,3,2,9] 
sorted_score_hat.sort(reverse=True)
sorted_inputscores = [1,4,3,4,7,3,9,5,100,4,2,6,8,2,9]
sorted_inputscores.sort(reverse=True)

print(sorted_score_hat)
print(sorted_inputscores)

print(round(0.45,1))