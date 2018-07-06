# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:26:28 2018

@author: Laura
"""

"""

parts of this code taken from https://github.com/MilkaLichtblau/FA-IR_Ranking/blob/master/src/main.py

"""

from src.algorithms.fair_ranker.FA_IR import fairRanking
#from utilsAndConstants.constants import ESSENTIALLY_ZERO

EVALUATE_FAILURE_PROBABILITY = 0



def runFAIR(dataSetName, protected, nonProtected, k):
    
        """
        Runs the FAIR ranking process
        
        @param dataSetName: the used data set with the ending 'pre' for preprocessed
        @param protected: list of protected candidates for the ranking
        @param nonProtected: list of non-protected candidates for the ranking
        @param k: truncation point of the ranking
        
        returns the final ranking, the candidates not selected for the ranking and the path for 
        CSV to save the ranking in
        """
        
        pairsOfPAndAlpha = initPAndAlpha(k)
        
        ranking, notSelected, nDataSetName = rankFAIR(protected, nonProtected, k, dataSetName, pairsOfPAndAlpha)
        
        rankingResultsPath = "FA-IR/" + nDataSetName + "ranking.csv"
        
        for candidate in ranking[:k]:
            candidate.qualification = candidate.learnedScores
        
        return ranking, notSelected, rankingResultsPath
        
def initPAndAlpha(k):
        """
        Initialization of p and alpha in accordance with the size of k
        input
        k      truncation point for ranking
        
        return list with pairs of p and alpha
        """
        if k == 40:
            
            pairsOfPAndAlpha = [(0.1, 0.1),  # no real results, skip in evaluation
                            (0.2, 0.1),  # no real results, skip in evaluation
                            (0.3, 0.1),  # no real results, skip in evaluation
                            (0.4, 0.1),  # no real results, skip in evaluation
                            (0.5, 0.0313),
                            (0.6, 0.0321),
                            (0.7, 0.0293),
                            (0.8, 0.0328),
                            (0.9, 0.0375)]
    
        elif k == 100:
            
            pairsOfPAndAlpha = [(0.1, 0.1),  # no real results, skip in evaluation
                                (0.2, 0.1),  # no real results, skip in evaluation
                                (0.3, 0.0220),
                                (0.4, 0.0222),
                                (0.5, 0.0207),
                                (0.6, 0.0209),
                                (0.7, 0.0216),
                                (0.8, 0.0216),
                                (0.9, 0.0256)]
        elif k == 1000:
            
            pairsOfPAndAlpha = [(0.1, 0.0140),
                            (0.2, 0.0115),
                            (0.3, 0.0103),
                            (0.4, 0.0099),
                            (0.5, 0.0096),
                            (0.6, 0.0093),
                            (0.7, 0.0094),
                            (0.8, 0.0095),
                            (0.9, 0.0100)]
            
        elif k == 1500:
            
            pairsOfPAndAlpha = [(0.1, 0.0122),
                            (0.2, 0.0101),
                            (0.3, 0.0092),
                            (0.4, 0.0088),
                            (0.5, 0.0084),
                            (0.6, 0.0085),
                            (0.7, 0.0084),
                            (0.8, 0.0084),
                            (0.9, 0.0096)]
    
        return pairsOfPAndAlpha
    
def calculateP(protected,nonProtected,dataSetName,k):
    
    """
    Calculates percentage of protected group in the whole data set
    
    @param protected: List with candidates that are members of protected group
    @param nonProtected: List with candidates that are members of unprotected group
    @param dataSetName: Name of the data set for evaluation
    @param k: truncation point of the ranking used to receive the appropriate value
    for p
    
    returns p the percentage of protected candidates in the data set
    """
    
    p = round(len(protected)/(len(protected)+(len(nonProtected))),1)
        
    #check if there is a real alpha value for p to prevent usage of dummy values
    if k == 40 and (p == 0.1 or p == 0.2 or p == 0.3 or p == 0.4):
        print("Info: Proportion in "+ dataSetName+" too small for k = 40. Using p = 0.5 instead.")
        p = 0.5
    elif k == 100 and (p == 0.1 or p == 0.2):
        print("Info: Proportion in "+ dataSetName+" too small for k = 100. Using p = 0.3 instead.")
        p = 0.3
        
    return p
    
def rankFAIR(protected, nonProtected, k, dataSetName, pairsOfPAndAlpha):
        """
        creates a fair ranking using FA*IR
        @param protected:        list of protected candidates, assumed to satisfy in-group monotonicty
        @param nonProtected:     list of non-protected candidates, assumed to satisfy in-group monotonicty
        @param k:                length of the rankings we want to create
        @param filePath:         directory in which to store the rankings
        @param pairsOfPAndAlpha: contains the mapping of a certain alpha correction to be used for a certain p
        The experimental setting is as follows: for a given data set of protected and non-

        return                   fair ranking and list of candidates that were not used in the ranking          
        """

        
        """
        print("colorblind ranking", end='', flush=True)
        colorblindRanking, colorblindNotSelected = fairRanking(k, protected, nonProtected, ESSENTIALLY_ZERO, 0.1)
        print(" [Done]")
        """
        
        #get p for the data set and the size k
        p = calculateP(protected,nonProtected,dataSetName,k)
        
        print("fair rankings", end='', flush=True)
        #create the fair ranking using FA*IR
        pair = [item for item in pairsOfPAndAlpha if item[0] == p][0]
        fairRankingOut, fairNotSelected = fairRanking(k, protected, nonProtected, pair[0], pair[1])
        print(" [Done]")
        
        
        return fairRankingOut, fairNotSelected, dataSetName