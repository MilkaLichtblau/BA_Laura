# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:12:43 2018

@author: Laura
"""

import math

def pak(k, ranking):
    
    """
    Calculates P@k 
    
    @param k: truncation point/length of the ranking
    @param ranking: list of candidates selected for the ranking
    
    return value for P@k
    """
    
    #cut the ranking at the given truncation point k
    pakRanking = ranking[:k]
    
    #initialize P@k
    pak = 0
    
    #check if rank in current ranking equals position in color-blind
    for i in range(k):
        if pakRanking[i].originalIndex == pakRanking[i].currentIndex:
            pak += 1
    
    #discount with truncation point
    pak = pak / k
    
    return pak
    
  
def ap(k, ranking):
    
    """
    Calculate AP

    @param k: truncation point/length of the ranking
    @param ranking: list of candidates selected for the ranking
    
    return AP
    """
    
    #initialize changing truncation point for calculation of P@k
    _k = 0
    
    #initialize AP
    ap = 0
    
    #check if rank in current ranking equals position in color-blind
    for i in range(k):
        _k += 1
        if ranking[i].originalIndex == ranking[i].currentIndex:
            ap += pak(_k, ranking)
       
    ap = ap / k
    
    return ap
    
def calculateMAP(dataSetName, results):
    """
    Calculate MAP
    
    @param dataSetName: Name of the data set
    @param results: List with results from earlier calculations
    
    return result list with the data set's name, the algorithm's name, MAP,
    and the value of MAP
    """
    #initialize variables
    qCount = 0
    mapColorBlind = 0
    mapFAIR = 0
    mapLFRanking = 0
    mapResults = []
    
    for i in range(len(results)):
        if dataSetName in results[i][0] and results[i][2] == 'AP':
            if results[i][1] == 'Color-Blind':
                mapColorBlind += results[i][3]
                #counts number of queries on one data set
                qCount +=1
            elif results[i][1] == 'FAIR':
                mapFAIR += results[i][3]
            elif results[i][1] == 'LFRanking':
                mapLFRanking += results[i][3]
    
    mapResults.append([dataSetName, 'Color-Blind', 'MAP', (mapColorBlind/qCount)])
    mapResults.append([dataSetName, 'FAIR', 'MAP', (mapFAIR/qCount)])
    mapResults.append([dataSetName, 'LFRanking', 'MAP', (mapLFRanking/qCount)])
    
    return mapResults
    

def nDCG(k, ranking, originalRanking):
    
    """
    Calculate NDCG
    
    @param k: truncation point/length of the ranking
    @param ranking: list of candidates selected for the ranking
    @param originalRanking: list of candidates from color-blind ranking
    
    return NDCG
    """
    
    #initialize params
    z = 0
    dCG = 0
    
    for i in range(k):
        z += (2**originalRanking[i].originalQualification - 1) / math.log((1 + i + 1),2)
        dCG += (2**ranking[i].originalQualification - 1) / math.log((1 + i + 1),2)
        
    nDCG = dCG / z
    
    return nDCG
        