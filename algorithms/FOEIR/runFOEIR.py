# -*- coding: utf-8 -*-
"""
Created on Fri May 25 18:46:35 2018

@author: Laura
"""

import numpy as np 
import math

def runFOEIR(ranking, k):
    
    scores = []
    ranks = []
    
    for candidate in ranking:
        scores.append(candidate.originalQualification)
        ranks.append(1 / math.log((1 + candidate.originalIndex + 1),2)
        
    u = np.asarray(scores)
    v = np.asarray(ranks)
    
    u.shape(len(u), 1)
    v.shape(1,len(v))
    
    uv = u*v
    
    