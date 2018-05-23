# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:16:56 2018

@author: Laura
"""

import math

def dTR(ranking, k):
    
    """
    Calculate Disparate Treatment Ratio (DTR)
    
    @param ranking: list with candidates from a given ranking
    @param k: truncation point/length of the given ranking
    
    return DTR
    """
    
    #initialize variables
    proUtility = 0
    unproUtility = 0
    proCount = 0
    unproCount = 0
    proExposure = 0
    unproExposure = 0
    
    #calculate utility and exposure for protected and unprotected groups
    for i in range(k):
        
        if ranking[i].isProtected == True:
            
            proCount += 1
            proUtility += ranking[i].originalQualification
            proExposure += 1/math.log((1+ranking[i].currentIndex),2)
            
        else:
            
            unproCount += 1
            unproUtility += ranking[i].originalQualification
            unproExposure += 1/math.log((1+ranking[i].currentIndex),2)

    #normalize with counter
    proUtility = proUtility / proCount
    unproUtility = unproUtility / unproCount          
    proExposure = proExposure / proCount
    unproExposure = unproExposure / unproCount
    
    #calculate DTR
    dTR = (proExposure / proUtility) / (unproExposure / unproUtility)
    
    return dTR

    
def dIR(ranking, k):
    
    """
    Calculate Disparate Impact Ratio (DIR)
    
    @param ranking: list with candidates from a given ranking
    @param k: truncation point/length of the given ranking
    
    return DIR
    """
    
    #initialize variables
    proUtility = 0
    unproUtility = 0
    proCount = 0
    unproCount = 0
    proCTR = 0
    unproCTR = 0
    
    #calculate utility and click through rate (CTR) for protected and unprotected groups
    for i in range(k):
        
        if ranking[i].isProtected == True:
            
            proCount += 1
            proUtility += ranking[i].originalQualification
            proCTR += ranking[i].originalQualification * (1/math.log((1+ranking[i].currentIndex),2))
            
        else:
            
            unproCount += 1
            unproUtility += ranking[i].originalQualification
            unproCTR += ranking[i].originalQualification * (1/math.log((1+ranking[i].currentIndex),2))

    #normalize with counter
    proUtility = proUtility / proCount
    unproUtility = unproUtility / unproCount          
    proCTR = proCTR / proCount
    unproCTR = unproCTR / unproCount
    
    #calculate DIR
    dIR = (proCTR / proUtility) / (unproCTR / unproUtility)
    
    return dIR
    
    
 