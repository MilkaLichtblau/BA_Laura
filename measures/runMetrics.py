# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:34:40 2018

@author: Laura
"""

import measures.calculaterKL as rKL
import measures.relevance as rel
import measures.calculateDTRandDIR as d

def runMetrics(k, protected, unprotected, ranking, originalRanking, dataSetName, algoName):
    
    """
    @param k: truncation point of the ranking
    @param user_N: length data set
    @param protected: list of candidate objects with membership of protected group
    @param unprotected: list of candidate objects with membership of non-protected group
    @param originRanking: list of candidates from the original data set ordered
    in decending order, hence providing a color-blind ranking of the size of the original
    data set
    @param ranking: list with candidates in the whole ranking
    @param dataSetName: String with the name of the data set
    @param algoName: String with the name of the algorithm that was used for the ranking
    
    return list with the name of the data set, the baselineAlgorithm, 
    """
    
    #initialize empty list for ranking indices
    indexRanking = []
    #initialize empty list for positive ranking indices
    proIndex = []
    #initialize empty list for original ranking scores with the lenght of the
    #data set
    originalScores = []
    #initialize empyt list for original scores in ranking
    originalScoresRanking = []
    
    #make sure originalRanking is still ordered color-blindly
    originalRanking.sort(key=lambda candidate: candidate.originalQualification, reverse=True)
    
    for j in range(len(originalRanking)):
        originalScores.append(originalRanking[j].originalQualification)
    
    #fill indexRanking and proIndex with values
    for i in range(len(ranking)):
        indexRanking.append(ranking[i].currentIndex)
        originalScoresRanking.append(ranking[i].originalQualification)
        if ranking[i].isProtected  == True:
            proIndex.append(ranking[i].originalIndex)
    
    #initialize length of originalRanking
    user_N = len(originalRanking)
    #initialize length of protected group
    pro_N = len(protected)
    #initialize result list
    results = []
    
    #calculate MAP      
    eval_AP = rel.ap(k, ranking)
    #append results
    results.append([dataSetName, algoName, 'AP', eval_AP])
    
    #calculate NDCG
    eval_NDCG = rel.nDCG(k, ranking, originalRanking)
    #append results
    results.append([dataSetName, algoName, 'NDCG', eval_NDCG])
    
    #calculate rKL
    #get the maximal rKL value
    max_rKL=rKL.getNormalizer(user_N,pro_N,dataSetName) 
    #get evaluation results for rKL
    eval_rKL=rKL.calculateNDFairness(indexRanking, proIndex, k, max_rKL)
    #append results
    results.append([dataSetName, algoName, 'rKL', eval_rKL])
    
    #calculate DTR
    eval_DTR = d.dTR(ranking, k)
    #append results
    results.append([dataSetName, algoName, 'DTR', eval_DTR])
    
    #calculate DTR
    eval_DIR = d.dIR(ranking, k)
    #append results
    results.append([dataSetName, algoName, 'DIR', eval_DIR])
    
    return results




