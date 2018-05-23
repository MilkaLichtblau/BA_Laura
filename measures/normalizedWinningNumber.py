# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:30:35 2018

@author: Laura
"""

ALGO_COLORBLIND = 'Color-Blind'
ALGO_FAIR = 'FAIR'
ALGO_LFRANING = 'LFRanking'

M_NDCG = 'NDCG'
M_RKL = 'rKL'
M_DTR = 'DTR'
M_DIR = 'DIR'
M_MAP = 'MAP'

def calculateNWN(dataSetName, results):
    
    for i in range(len(results)):
        if dataSetName in results[i][0] and results[i][2] == M_NDCG:
            if results[i][1] == :
                ndcgColorBlind += results[i][3]
                #counts number of queries on one data set
                qCount +=1
            elif results[i][1] == 'FAIR':
                ndcgFAIR += results[i][3]
            elif results[i][1] == 'LFRanking':
                ndcgLFRanking += results[i][3]