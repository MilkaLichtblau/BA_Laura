# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:30:35 2018

@author: Laura
"""
#constants for algorithms
ALGO_COLORBLIND = 'Color-Blind'
ALGO_FAIR = 'FAIR'
ALGO_LFRANING = 'LFRanking'

#constants for measures
M_NDCG = 'NDCG'
M_RKL = 'rKL'
M_DTR = 'DTR'
M_DIR = 'DIR'
M_MAP = 'MAP'

#constants for data sets
GERMANCREDIT = 'GermanCredit'
GC_AGE25 = 'GermanCredit_age35'
GC_AGE35 = 'GermanCredit_age35'
GC_SEX = 'GermanCredit_sex'

def calculateNWN(results):
    
    
    ndcgCount = 0
    rKLCount = 0
    dTRCount = 0
    dIRCount = 0
    mAPCount = 0
    ndcgVal = 0
    rKLVal = 0
    dTRVal = 0
    dIRVal = 0
    mAPVal = 0
    midResults = []
    
    dataSetList = [GERMANCREDIT, GC_AGE25, GC_AGE35, GC_SEX]
    algoList = [ALGO_COLORBLIND,ALGO_FAIR,ALGO_LFRANING]
    
    #calculate Winning Number
    for d in dataSetList:
        #get only data from one data set at a time
        dataList = createDataSetList(d, results)
        for algo in algoList:
            #get data from one algorithm at a time
            algoMeasureList = createAlgoList(algo, dataList)
            for m in algoMeasureList: 
                #check if ndcg
                if m[2] == M_NDCG:
                    #compare with other algorithms evaluated for this data set
                    for runningAlgo in dataList:
                        if m[3] > runningAlgo[3]:
                            ndcgVal += 1
                #check if rKL
                elif m[2] == M_RKL:
                    #compare with other algorithms evaluated for this data set
                    for runningAlgo in dataList:
                        if m[3] < runningAlgo[3]:
                            rKLVal += 1
                #check if DTR
                elif m[2] == M_DTR:
                    #compare with other algorithms evaluated for this data set
                    for runningAlgo in dataList:
                        if dist(m[3]) < dist(runningAlgo[3]):
                            dTRVal += 1
                #check if DIR
                elif m[2] == M_DIR:
                    #compare with other algorithms evaluated for this data set
                    for runningAlgo in dataList:
                        if dist(m[3]) < dist(runningAlgo[3]):
                            dIRVal += 1
                #check if MAP
                elif m[2] == M_MAP:
                    #compare with other algorithms evaluated for this data set
                    for runningAlgo in dataList:
                        if m[3] > runningAlgo[3]:
                            mAPVal += 1
            midResults.append([algo, ndcgVal, rKLVal, dTRVal, dIRVal, mAPVal])
            
    print(midResults)
    

def dist(val):
    
    if val < 1:
        return 1 - val
    else:
        return val - 1          
            
def createDataSetList(dataSetName, results):
    
    dataList = []
    
    for i in results:
        if dataSetName in i[0]:
            dataList.append(i)
            
    return dataList

def createAlgoList(algoName, results):
    
    dataList = []
    
    for i in results:
        if algoName == i[1]:
            dataList.append(i)
            
    return dataList
    