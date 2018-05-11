# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:33:08 2018

@author: Laura
"""

import csv
import numpy as np

def readFile(filepath):
 
    with open(filepath, newline='') as File:  
        reader = csv.reader(File)
        a = np.array([row for row in reader])
    return a
        
def writeFile(filepath, data):
    
    myFile = open(filepath, 'w', newline='')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)
         
    print("Writing complete")
