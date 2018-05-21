# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:34:40 2018

@author: Laura
"""

import measures.calculaterKL as rKL

def runMetrics(k, user_N, protected, unprotected, ranking, dataSetName, algoName):
    
    """
    @param k: truncation point of the ranking
    @param user_N: length data set
    @param protected: list of candidate objects with membership of protected group
    @param unprotected: list of candidate objects with membership of non-protected group
    @param ranking: list with candidates in the whole ranking
    @param dataSetName: String with the name of the data set
    @param algoName: String with the name of the algorithm that was used for the ranking
    
    return list with the name of the data set, the baselineAlgorithm, 
    """
    
    #initialize empty list for ranking indices
    indexRanking = []
    
    #initialize empty list for positive ranking indices
    proIndex = []
    
    #fill lists with values
    for i in range(len(ranking)):
        indexRanking.append(ranking[i].currentIndex)
        if ranking[i].isProtected  == True:
            proIndex.append(ranking[i].originalIndex)
    
    #initialize length of protected group
    pro_N = len(protected)
    
    max_rKL=rKL.getNormalizer(user_N,pro_N,dataSetName) 
    
    eval_rKL=rKL.calculateNDFairness(indexRanking, proIndex, k, max_rKL)
    
    results = [dataSetName, algoName, 'rKL', eval_rKL]
    
    return results




