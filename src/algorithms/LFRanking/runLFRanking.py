# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:22:43 2018

@author: Laura
"""

from __future__ import division
import scipy.optimize as optim
import numpy as np
from src.algorithms.LFRanking import LearningFairRankingOptimization
from src.algorithms.LFRanking import utility # import for calculation of weighted scores

"""
Parts of this code taken from https://github.com/DataResponsibly/FairRank/blob/master/optimization.py
and https://github.com/DataResponsibly/FairRank/blob/master/runOptimization.py
"""


def runLFRanking(ranking,protected,unprotected,_k,DataSetName):

    """
        Run the optimization process.
        @param ranking: list with candidates that can be used for the ranking
        @param protected: list with candidates belonging to the protected group
        @param unrpotected: list with candidates belonging to the unprotected group   
        @param _k: The number of clusters in the intermediate layer of neural network
        
        returns the final ranking, the candidates not selected for the ranking and the path for 
        CSV to save the ranking in
    """        
    
    dat = []
    pro_dat = []
    unpro_dat = []
    scores = []
    
    ranking.sort(key=lambda candidate: candidate.learnedScores, reverse=True)
    
    for i in range(len(ranking)):
        dat.append(ranking[i].features)
        scores.append(ranking[i].learnedScores)
        
    protected.sort(key=lambda candidate: candidate.learnedScores, reverse=True)
    
    for i in range(len(protected)):
        pro_dat.append(protected[i].features)
        
    unprotected.sort(key=lambda candidate: candidate.learnedScores, reverse=True)
        
    for i in range(len(unprotected)):
        unpro_dat.append(unprotected[i].features)
    
    data = np.asarray(dat)
    input_scores= np.asarray(scores)
    pro_data = np.asarray(pro_dat)
    unpro_data = np.asarray(unpro_dat)
    
    print("start LFRanking opt")

    # initialize the optimization
    rez,bnd=LearningFairRankingOptimization.initOptimization(data,_k) 
    
    LearningFairRankingOptimization.lbfgsOptimize.iters=0                
    rez = optim.fmin_l_bfgs_b(LearningFairRankingOptimization.lbfgsOptimize, x0=rez, disp=1, epsilon=1e-5, 
                   args=(data, pro_data, unpro_data, input_scores, _k, 0.01,
                         1, 100, 0), bounds = bnd,approx_grad=True, factr=1e12, pgtol=1e-04,maxfun=15000, maxiter=15000)


    print ("Ending LFRanking optimization")
    # evaluation after converged
    
    user_N,att_N=data.shape
    
    # initialize the clusters
    clusters=np.matrix(rez[0][(2 * att_N) + _k:]).reshape((_k, att_N))
    
    # get the distance between input user X and intermediate clusters Z
    dists_x = LearningFairRankingOptimization.distances(data, clusters, user_N, att_N, _k)
    # compute the probability of each X maps to Z
    Mnk_x= LearningFairRankingOptimization.M_nk(dists_x, user_N, _k)
    
    fairRanking = calculateFinalEstimateY(Mnk_x, input_scores, clusters, user_N, _k, ranking)
    
    rankingResultsPath = "LFRanking/" + DataSetName + "ranking.csv"
        
    return fairRanking, rankingResultsPath
 
    
def calculateFinalEstimateY(_M_nk_x, _inputscores, _clusters, _N, _k, ranking):
    """
        Calculate the estimated score and ranking accuracy of corresponding ranking.
        :param _M_nk_x: The probability mapping matrix from input X and clusters Z 
        :param _inputscores: The input scores of all users
        :param _clusters: The clusters in the intermediate Z
        :param _N: The total user number in input X        
        :param _k: The number of clusters in the intermediate layer of neural network 
        @param ranking: list of candidates for the ranking
        :return: returns the estimated X and loss between input X and estimated X.
    """
    score_hat = np.zeros(_N) # initialize the estimated scores
    
    # calculate estimate score of each user by mapping probability between X and Z     
    for ui in range(_N):
        score_hat_u = 0.0
        for ki in range(_k):
            score_hat_u += (_M_nk_x[ui,ki] * _clusters[ki])                         
        score_hat[ui] = utility.calculateWeightedScores(score_hat_u)
    
    score_hat=list(score_hat)
    
    # sort the scores in descending order
    for i in range(len(ranking)):
        ranking[i].qualification = score_hat[i]    
     
    # order ranking according to new scores
    ranking.sort(key=lambda candidate: candidate.qualification, reverse=True)
    
    return ranking