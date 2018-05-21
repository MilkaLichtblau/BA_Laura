# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:12:43 2018

@author: Laura
"""

import numpy as np

def binaryLabels(dataSetLength, scores, k):
    
    """
    Calculates the score at which an item becomes irrelevant. We decide that
    by a percentile at a certain point and all scores above that percentile
    are relevant and all scores beneath are irrelevant. We calculate the 
    point for the percentile as follows: 1 - ((k/2)/dataSetLength)
    
    @param dataSetLenght: Lenght of the original data set
    @param scores: Scores from the original ranking ordered color-blindly
    @param k: Truncation point/length of the actual ranking
    """
    
    scoreArray = np.array(scores)
    
    #calculate Percentile
    q = 1 - ((k/2)/dataSetLength)
    q = round(q,2)
    q = q * 100
    
    p = np.percentile(scoreArray, q)
    
    return p
    
def mAP(p, rankingLength, scores):
    
    scoreArray = np.array(scores)
    
    
