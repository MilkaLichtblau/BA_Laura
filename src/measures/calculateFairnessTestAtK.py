# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:30:09 2018

@author: Laura
"""

from src.algorithms.fair_ranker.test import FairnessInRankingsTester
from src.algorithms.fair_ranker.runRankFAIR import initPAndAlpha, calculateP

def fairnessTestAtK(dataSetName, ranking, protected, unProtected, k):
    
    """
    Calculates at which prefix the ranking starts to be unfair with respect to 
    the proportion of the protected group in the ranking. We use the statistical
    test used for FA*IR to receive that prefix. We then normalize the prefix
    with respect to the size of the given ranking (k). We will refer to that
    measure as FairnessAtK.
    
    @param dataSetName: Name of the data set, used to notify the user for which
    data set a bigger p is needed if the proportions for that are too small.
    @param ranking: list with candidates in the whole ranking
    @param protected: list of candidate objects with membership of protected group
    from the original data set
    @param unprotected: list of candidate objects with membership of non-protected group
    from the original data set
    @param k: truncation point/length of the ranking
    
    return the value for FairnessAtK
    """
    
    ranking = ranking[:k]
    
    #initialize p and alpha values for given k
    pairsOfPAndAlpha = initPAndAlpha(k)
    
    #calculates the percentage of protected items in the data set
    p = calculateP(protected,unProtected,dataSetName,k)
    
    pair = [item for item in pairsOfPAndAlpha if item[0] == p][0]
    
    #initialize a FairnessInRankingsTester object
    gft = FairnessInRankingsTester(pair[0], pair[1], k, correctedAlpha=True)
    
    #get the index until the ranking can be considered as fair, m will equal true if the whole set is true
    t, m = FairnessInRankingsTester.ranked_group_fairness_condition(gft, ranking)
    
    if m == False:
        #calculate and normalize Fairness@k
        return t/len(ranking)
    else:
        #return 1.0 if everything is fair
        return 1.0
    
    