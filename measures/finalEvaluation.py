# -*- coding: utf-8 -*-
"""
Created on Sat May 26 14:01:41 2018

@author: Laura

"""

from collections import Counter

#constants for algorithms
ALGO_COLORBLIND = 'Color-Blind'
ALGO_FAIR = 'FAIR'
ALGO_LFRANING = 'LFRanking'
ALGO_FELDMAN = 'FeldmanEtAl'

#constants for measures
M_NDCG = 'NDCG'
M_RKL = 'rKL'
M_DTR = 'DTR'
M_DIR = 'DIR'
M_MAP = 'MAP'

def calculateFinalEvaluation(results, fileNames):
    
    algoList = [ALGO_COLORBLIND,ALGO_FAIR,ALGO_LFRANING,ALGO_FELDMAN]
    
    dataSets = []
    querySets = []
    
    for name in fileNames:
        if '_' not in name:
            name = name.split('.')
            dataSets.append(name[0])
        else:
            name = name.split('_')
            querySets.append(name[0])
      
    #getting unique elements
    querySets = set(querySets)
    querySets = list(querySets)
    
    resultsForWN = []
    finalResults = []
    helperWinningAlgoList = []
    helperAlgoCount = []
    winningAlgoList = []
    algoCount = []
    
    for name in dataSets:
        resultsForWN = []
        for row in results:
            if name in row[0]:
                #since MAP is the average AP over multiple queries and 
                #this is only one data set and not multiple queries
                #MAP is equal to AP 
                if 'AP' == row[2]:
                    row[2] = 'MAP'
                finalResults.append(row)
                resultsForWN.append(row[1:])
        helperWinningAlgoList, helperAlgoCount = getWinningAlgoForMeasureInDataSet(resultsForWN, name, algoList)
        winningAlgoList += helperWinningAlgoList
        algoCount += helperAlgoCount
    
    print(winningAlgoList)
    #print(algoCount)
    
    #print(calculateOccurence(algoCount,algoList) )

    
def calculateOccurence(algoCount,algoList):
        
    comNDCG = []
    comRKL = []
    comDTR = []
    comDIR = []
    comMAP = []
    
    for row in algoCount:
        if row[1] == M_NDCG:
            comNDCG.append(row[0])         
        #check if rKL
        elif row[1] == M_RKL:
            comRKL.append(row[0])
        #check if DTR
        elif row[1] == M_DTR:
            comDTR.append(row[0])
        #check if DIR
        elif row[1] == M_DIR:
            comDIR.append(row[0])
        #check if MAP
        elif row[1] == M_MAP:
            comMAP.append(row[0])
            
    return Counter(comNDCG), Counter(comRKL), Counter(comDTR), Counter(comDIR), Counter(comMAP)
        
def getWinningAlgoForMeasureInDataSet(resultsForWN,name,algoList):
    
    comNDCG = []
    comRKL = []
    comDTR = []
    comDIR = []
    comMAP = []
    
    algoCount = []
    midResults = []
    
    for row in resultsForWN:
        #check if ndcg
        if row[1] == M_NDCG:
            #save data for comparison
            comNDCG.append(row)
            for algo in algoList:
                if algo == row[0]:
                    algoCount.append([algo, M_NDCG])
                
        #check if rKL
        elif row[1] == M_RKL:
            #save data for comparison
            comRKL.append(row)
            for algo in algoList:
                if algo == row[0]:
                    algoCount.append([algo, M_RKL])
        #check if DTR
        elif row[1] == M_DTR:
            #save data for comparison
            comDTR.append(row)
            for algo in algoList:
                if algo == row[0]:
                    algoCount.append([algo, M_DTR])
        #check if DIR
        elif row[1] == M_DIR:
            #save data for comparison
            comDIR.append(row)
            for algo in algoList:
                if algo == row[0]:
                    algoCount.append([algo, M_DIR])
        #check if MAP
        elif row[1] == M_MAP:
            #save data for comparison
            comMAP.append(row) 
            for algo in algoList:
                if algo == row[0]:
                    algoCount.append([algo, M_MAP])
    
    midResults+=(compareNDCG(comNDCG, algoList))
    midResults+=(compareRKL(comRKL, algoList))
    midResults+=(compareDTR(comDTR, algoList))
    midResults+=(compareDIR(comDIR, algoList))
    midResults+=(compareMAP(comMAP, algoList))
    
    return midResults, algoCount
    
def compareNDCG(compareList, algoList):
    
    results = []
    algoCount = 0
    
    for algo in algoList:
        algoCount = 0
        for row in compareList:
            if algo == row[0]:
                value = row[2]
                for row in compareList:
                    if value > row[2]:
                        algoCount += 1
        results.append([algo, M_NDCG, algoCount])
        
    return results

def compareRKL(compareList,algoList):
    
    results = []
    algoCount = 0
    
    for algo in algoList:
        algoCount = 0
        for row in compareList:
            if algo == row[0]:
                value = row[2]
                for row in compareList:
                    if value < row[2]:
                        algoCount +=1
        results.append([algo, M_RKL, algoCount])
        
    return results

def compareDTR(compareList, algoList):
    
    results = []
    algoCount = 0
    
    for algo in algoList:
        algoCount = 0
        for row in compareList:
            if algo == row[0]:
                value = row[2]
                for row in compareList:
                    if dist(value) < dist(row[2]):
                        algoCount+= 1
        results.append([algo, M_DTR, algoCount])
    return results
                    
def compareDIR(compareList, algoList):
    
    results = []
    algoCount = 0
    
    for algo in algoList:
        algoCount = 0
        for row in compareList:
            if algo == row[0]:
                value = row[2]
                for row in compareList:
                    if dist(value) < dist(row[2]):
                        algoCount+=1
        results.append([algo, M_DIR,algoCount])
    return results
    
def compareMAP(compareList, algoList):
    
    results = []
    algoCount = 0
    
    for algo in algoList:
        algoCount = 0
        for row in compareList:
            if algo == row[0]:
                value = row[2]
                for row in compareList:
                    if value > row[2]:
                        algoCount +=1
        results.append([algo, M_MAP,algoCount])
    return results
                    
def dist(val):
    
    if val < 1:
        return 1 - val
    else:
        return val - 1                
                
                
    
results = [['GermanCreditAge25', 'Color-Blind', 'AP', 1.0], ['GermanCreditAge25', 'Color-Blind', 'NDCG', 1.0], ['GermanCreditAge25', 'Color-Blind', 'rKL', 0.0019415156418714987], ['GermanCreditAge25', 'Color-Blind', 'DTR', 0.9513075970964544], ['GermanCreditAge25', 'Color-Blind', 'DIR', 0.9394417693807077], ['GermanCreditAge25', 'FeldmanEtAl', 'AP', 1.0], ['GermanCreditAge25', 'FeldmanEtAl', 'NDCG', 0.9940294681008911], ['GermanCreditAge25', 'FeldmanEtAl', 'rKL', 0.0017166872359973236], ['GermanCreditAge25', 'FeldmanEtAl', 'DTR', 0.8508150326312649], ['GermanCreditAge25', 'FeldmanEtAl', 'DIR', 0.8109361787503478], ['GermanCreditAge25', 'FAIR', 'AP', 0.00774059274059274], ['GermanCreditAge25', 'FAIR', 'NDCG', 0.9898008868890052], ['GermanCreditAge25', 'FAIR', 'rKL', 0.001685462935765634], ['GermanCreditAge25', 'FAIR', 'DTR', 1.1557388518132612], ['GermanCreditAge25', 'FAIR', 'DIR', 1.0940549790492724], ['GermanCreditAge25', 'LFRanking', 'AP', 1.0], ['GermanCreditAge25', 'LFRanking', 'NDCG', 1.0], ['GermanCreditAge25', 'LFRanking', 'rKL', 0.0019415156418714987], ['GermanCreditAge25', 'LFRanking', 'DTR', 0.9513075970964544], ['GermanCreditAge25', 'LFRanking', 'DIR', 0.9394417693807077]] #['GermanCreditAge35', 'Color-Blind', 'AP', 1.0], ['GermanCreditAge35', 'Color-Blind', 'NDCG', 1.0], ['GermanCreditAge35', 'Color-Blind', 'rKL', 0.013221817354072955], ['GermanCreditAge35', 'Color-Blind', 'DTR', 1.0254952326065843], ['GermanCreditAge35', 'Color-Blind', 'DIR', 1.0294769735762885], ['GermanCreditAge35', 'FeldmanEtAl', 'AP', 1.0], ['GermanCreditAge35', 'FeldmanEtAl', 'NDCG', 0.9821805792350532], ['GermanCreditAge35', 'FeldmanEtAl', 'rKL', 0.016581041467974518], ['GermanCreditAge35', 'FeldmanEtAl', 'DTR', 0.7609134808537342], ['GermanCreditAge35', 'FeldmanEtAl', 'DIR', 0.7061008645947737], ['GermanCreditAge35', 'FAIR', 'AP', 0.0], ['GermanCreditAge35', 'FAIR', 'NDCG', 0.9821805792350532], ['GermanCreditAge35', 'FAIR', 'rKL', 0.00520983810847107], ['GermanCreditAge35', 'FAIR', 'DTR', 1.1083207622873845], ['GermanCreditAge35', 'FAIR', 'DIR', 1.0406462482475403], ['GermanCreditAge35', 'LFRanking', 'AP', 1.0], ['GermanCreditAge35', 'LFRanking', 'NDCG', 1.0], ['GermanCreditAge35', 'LFRanking', 'rKL', 0.013221817354072955], ['GermanCreditAge35', 'LFRanking', 'DTR', 1.0254952326065843], ['GermanCreditAge35', 'LFRanking', 'DIR', 1.0294769735762885], ['GermanCreditSex', 'Color-Blind', 'AP', 1.0], ['GermanCreditSex', 'Color-Blind', 'NDCG', 1.0], ['GermanCreditSex', 'Color-Blind', 'rKL', 0.0008560666111084664], ['GermanCreditSex', 'Color-Blind', 'DTR', 1.0522788238170926], ['GermanCreditSex', 'Color-Blind', 'DIR', 1.066408846804308], ['GermanCreditSex', 'FeldmanEtAl', 'AP', 1.0], ['GermanCreditSex', 'FeldmanEtAl', 'NDCG', 0.9985107454401517], ['GermanCreditSex', 'FeldmanEtAl', 'rKL', 0.013239711180642978], ['GermanCreditSex', 'FeldmanEtAl', 'DTR', 1.1105073352926267], ['GermanCreditSex', 'FeldmanEtAl', 'DIR', 1.1413786061494602], ['GermanCreditSex', 'FAIR', 'AP', 0.02660704156954157], ['GermanCreditSex', 'FAIR', 'NDCG', 0.9985107454401517], ['GermanCreditSex', 'FAIR', 'rKL', 0.0], ['GermanCreditSex', 'FAIR', 'DTR', 0.9862620820134426], ['GermanCreditSex', 'FAIR', 'DIR', 1.0091629255201566], ['GermanCreditSex', 'LFRanking', 'AP', 1.0], ['GermanCreditSex', 'LFRanking', 'NDCG', 1.0], ['GermanCreditSex', 'LFRanking', 'rKL', 0.0008560666111084664], ['GermanCreditSex', 'LFRanking', 'DTR', 1.0522788238170926], ['GermanCreditSex', 'LFRanking', 'DIR', 1.066408846804308]]


calculateFinalEvaluation(results, ['GermanCreditAge25'])