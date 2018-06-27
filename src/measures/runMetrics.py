# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:34:40 2018

@author: Laura
"""

import src.measures.calculaterKL as rKL
import src.measures.relevance as rel
import src.measures.calculateDTRandDIR as d
import src.measures.calculateFairnessTestAtK as ftak
import copy

"""
This method runs the evaluation process.
Except for the three NDCG values all values are evaluated for the given size of
k. 
"""

def runMetrics(k, protected, unprotected, ranking, originalRanking, dataSetName, algoName):
    
    """
    Starts the evaluation process for on measures for the inputed data
    
    @param k: truncation point of the ranking
    @param user_N: length data set
    @param protected: list of candidate objects with membership of protected group
    from the original data set
    @param unprotected: list of candidate objects with membership of non-protected group
    from the original data set
    @param originRanking: list of candidates from the original data set ordered
    in decending order, hence providing a color-blind ranking of the size of the original
    data set
    @param ranking: list with candidates in the whole ranking
    @param dataSetName: String with the name of the data set
    @param algoName: String with the name of the algorithm that was used for the ranking
    
    return list with the name of the data set, the baselineAlgorithm, 
    """
    
    #checking k for Fairness@k, currently only evaluation of 40, 100, 1000, or 1500 possible
    if k != 40 and k != 100 and k != 1000 and k != 1500:
        print('Cannot obtain alpha adjustment, for k='+str(k)+ 'Setting it to 40 as default.')
        k = 40
    
    #initialize empty list for ranking indices
    indexRanking = []
    #initialize empty list for positive ranking indices
    proIndex = []
    
    #deep copy since we need to sort this list differently from the other ranking list
    oR = copy.deepcopy(originalRanking)
    
    #make sure ranking is sorted according to its qualifications
    ranking.sort(key=lambda candidate: candidate.qualification, reverse=True)
    
    #make sure originalRanking is still ordered descendingly on original qualifications
    oR.sort(key=lambda candidate: candidate.originalQualification, reverse=True)
    
    #fill indexRanking and proIndex with values
    for i in range(len(ranking)):
        indexRanking.append(ranking[i].currentIndex)
        if ranking[i].isProtected  == True:
            proIndex.append(ranking[i].originalIndex)
    
    #initialize length of originalRanking
    user_N = len(oR)
    #initialize length of protected group
    pro_N = len(protected)
    #initialize result list
    results = []
    
    #calculate MAP      
    eval_AP = rel.ap(k, ranking)
    #append results
    results.append([dataSetName, algoName, 'AP', eval_AP])
    
    #calculate NDCG@1
    eval_NDCG = rel.nDCG(1, ranking, oR)
    #append results
    results.append([dataSetName, algoName, 'NDCG@1', eval_NDCG])
    
    #calculate NDCG@5
    eval_NDCG = rel.nDCG(5, ranking, oR)
    #append results
    results.append([dataSetName, algoName, 'NDCG@5', eval_NDCG])
    
    #calculate NDCG@10
    eval_NDCG = rel.nDCG(10, ranking, oR)
    #append results
    results.append([dataSetName, algoName, 'NDCG@10', eval_NDCG])
    
    #calculate rKL
    #get the maximal rKL value
    max_rKL=rKL.getNormalizer(user_N,pro_N,dataSetName) 
    #get evaluation results for rKL
    eval_rKL=rKL.calculateNDFairness(indexRanking, proIndex, k, max_rKL)
    #append results
    results.append([dataSetName, algoName, 'rKL', eval_rKL])
    
    #calculate DTR and DIR and return their results
    results += d.calculatedTRandDIR(ranking, algoName, dataSetName, k = 100)
    
    #calculate Fairnes@k
    eval_FairnessAtK = ftak.fairnessTestAtK(dataSetName, ranking, protected, unprotected, k)
    
    results.append([dataSetName, algoName, 'FairnessAtK', eval_FairnessAtK])
    
    return results




