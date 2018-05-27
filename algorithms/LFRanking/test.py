# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:08:41 2018

@author: Laura
"""
import csv
import os
"""
Creates multiple CSV files each one storing a query from a raw CSV with 
multiple queries inside. 
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

