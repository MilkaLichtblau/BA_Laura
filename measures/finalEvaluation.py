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
M_AP = 'AP'

def calculateFinalEvaluation(results, fileNames):
    
    """
    Calculates the final result list with Normalized Winning Number (NWN)
    
    @param results: results list from the evaluation of the measures
    looks like [dataSetName, algorithmName, measureName, measureValue]
    @param fileNames: Names from the evaluation files, we assume that 
    data set names do not include any underscores (_) because we identify
    file names with underscores as queries and will evalute average results for
    those
    
    return final results for the evaluation formatted as input list, however,
    since NWN is evaluated across all data sets, we substituted the data set
    name with NWN making those results look as follows:
        ['NWN', algorithmName, measureName, valueOfNWNforMeasureAndAlgo]
    """
    
    algoList = [ALGO_COLORBLIND,ALGO_FAIR,ALGO_LFRANING,ALGO_FELDMAN]
    
    dataSets = []
    querySets = []
    
    resultsForWN = []
    midResults = []
    finalResults = []
    
    lNDCG = []
    lRKL = []
    lDTR = []
    lDIR = []
    lMAP = []
    lAP = []
    
    for name in fileNames:
        if '_' not in name:
            name = name.split('.')
            dataSets.append(name[0])
        else:
            name = name.split('.')
            querySets.append(name[0])
            name = name[0].split('_')
            dataSets.append(name[0])
            
    dataSetDic = Counter(dataSets)
    
    #getting unique elements
    dataSets = set(dataSets)
    dataSets = list(dataSets)
    
    
    for name in dataSets:
        helpResults = []
        for row in results:
            if name == row[0]:
                midResults.append(row)
            elif name in row[0]:
                helpResults.append(row[1:])
        if dataSetDic[name] != 1:
            hNDCG, hRKL, hDTR, hDIR, hMAP, hAP = getListForMeasureInDataSet(helpResults, algoList)
            lNDCG += hNDCG
            lRKL += hRKL
            lDTR += hDTR
            lDIR += hDIR
            lAP += hAP
            midResults += calculateAverage(lNDCG, lRKL, lDTR, lDIR, lAP, dataSetDic, name, algoList)
        
    
        
    for name in querySets:
        helpResults = []
        for row in results:
            if name != row[0]:
                midResults.append(row)
            elif name == row[0]:
                helpResults.append(row[1:])
                hNDCG, hRKL, hDTR, hDIR, hMAP, hAP = getListForMeasureInDataSet(helpResults, algoList)
                lNDCG += hNDCG
                lRKL += hRKL
                lDTR += hDTR
                lDIR += hDIR
                lMAP += hMAP
                
    
    lNDCG = []
    lRKL = []
    lDTR = []
    lDIR = []
    lMAP = []
    
    #get WN for each data set
    for name in dataSets:
        #reset result list for each data set
        resultsForWN = []
        #run through the result list from a data set
        for row in midResults:
            if name == row[0]:
                #since MAP is the average AP over multiple queries and 
                #this is only one data set and not multiple queries
                #MAP is equal to AP 
                if M_AP == row[2]:
                    row[2] = M_MAP
                #append rows from the data set to the final result list    
                finalResults.append(row)
                #get rows for calculation of NWN
                resultsForWN.append(row[1:])
        #get the count for how often one algorithm won against all other algorithms
        #also get IWN once appendet in the list for hNDCG since we always evaluate
        #all measures, hence IWN will be the same for all measures if an evaluation for
        #the data set occured
        hNDCG, hRKL, hDTR, hDIR, hMAP, hAP = getListForMeasureInDataSet(resultsForWN, algoList)
        hNDCG, hRKL, hDTR, hDIR, hMAP = compareMeasures(hNDCG, hRKL, hDTR, hDIR, hMAP,algoList)
        lNDCG += hNDCG
        lRKL += hRKL
        lDTR += hDTR
        lDIR += hDIR
        lMAP += hMAP
    
    finalResults += calculateNWN(lNDCG, lRKL, lDTR, lDIR, lMAP, algoList)
    
    return finalResults
    
    
def calculateAverage(lNDCG, lRKL, lDTR, lDIR, lAP, dataSetDic, dataSetName, algoList):
    
    """
    Calculates the average values over a set of queries for each measure and algorithm
    
    @param lNDCG: list with WN for all algorithms for NDCG plus the IWN
    @param lRKL: list with WN for all algorithms for rKL
    @param lDTR: list with WN for all algorithms for DTR
    @param lDIR: list with WN for all algorithms for DIR
    @param lMAP: list with WN for all algorithms for MAP
    @param dataSetDic: A dictionary with the data set name and the total number
    queries for that data set
    @param dataSetName: Name of data set
    @param algoList: List with all algorithms in the benchmarking
    
    return result list with average values for each measure and each algorithm
    """
    
    actualAlgoList = []
    results = []
    
    #check whih algorithms were evaluated on this data set
    for algo in algoList:
        for row in lNDCG:
            if algo == row[0]:
                #append actually included algorithms for this data set
                actualAlgoList.append(algo)
                
    
    #make actualAlgoList a list of unique items
    actualAlgoList = set(actualAlgoList)
    actualAlgoList = list(actualAlgoList)
    
    
    for algo in actualAlgoList:
        resNDCG = 0
        resRKL = 0
        resDTR = 0
        resDIR = 0
        resAP = 0
        for nDCG, rKL, dTR, dIR, aP in zip(lNDCG, lRKL, lDTR, lDIR, lAP):
            if algo == nDCG[0]:
                resNDCG += nDCG[2]
            if algo == rKL[0]:
                resRKL += rKL[2]
            if algo == dTR[0]:
                resDTR += dTR[2]
            if algo == dIR[0]:
                resDIR += dIR[2]
            if algo == aP[0]:
                resAP += aP[2]
        results.append([dataSetName, algo, M_NDCG, resNDCG/dataSetDic[dataSetName]])
        results.append([dataSetName, algo, M_MAP, resAP/dataSetDic[dataSetName]])
        results.append([dataSetName, algo, M_RKL, resRKL/dataSetDic[dataSetName]]) 
        results.append([dataSetName, algo, M_DTR, resDTR/dataSetDic[dataSetName]]) 
        results.append([dataSetName, algo, M_DIR, resDIR/dataSetDic[dataSetName]]) 
        
        
    return results
    
  
def calculateNWN(lNDCG, lRKL, lDTR, lDIR, lMAP, algoList):
        
    """
    @param lNDCG: list with WN for all algorithms for NDCG plus the IWN
    @param lRKL: list with WN for all algorithms for rKL
    @param lDTR: list with WN for all algorithms for DTR
    @param lDIR: list with WN for all algorithms for DIR
    @param lMAP: list with WN for all algorithms for MAP
    @param algoList: List with all algorithms in the benchmarking
    
    returns list with NWN for each algorithm on each measure
    """
    
    
    resNDCG = 0
    resRKL = 0
    resDTR = 0
    resDIR = 0
    resMAP = 0
    
    idealWinningNumber = 0
    
    resultsNWN = []
    
    for algo in algoList:
        resNDCG = 0
        resRKL = 0
        resDTR = 0
        resDIR = 0
        resMAP = 0
        idealWinningNumber = 0
        for nDCG, rKL, dTR, dIR, mAP in zip(lNDCG, lRKL, lDTR, lDIR, lMAP):
            if algo == nDCG[0]:
                resNDCG += nDCG[2]
                idealWinningNumber += nDCG[3]
            if algo == rKL[0]:
                resRKL += rKL[2]
            if algo == dTR[0]:
                resDTR += dTR[2]
            if algo == dIR[0]:
                resDIR += dIR[2]
            if algo == mAP[0]:
                resMAP += mAP[2]
        resultsNWN.append(['NWN', algo, M_NDCG, resNDCG/idealWinningNumber])
        resultsNWN.append(['NWN', algo, M_MAP, resMAP/idealWinningNumber])
        resultsNWN.append(['NWN', algo, M_RKL, resRKL/idealWinningNumber])
        resultsNWN.append(['NWN', algo, M_DTR, resDTR/idealWinningNumber])
        resultsNWN.append(['NWN', algo, M_DIR, resDIR/idealWinningNumber])

        
    return resultsNWN
       
def getListForMeasureInDataSet(resultsForWN, algoList):
    
    """
    Get result lists for each measure in a data set
    
    @param resultsForWN: list with results from the evaluation measures on one
    data set
    @param algoList: List with all algorithms in the benchmarking
    
    return lists for each measure with its evaluation results from the corresponding methods
    """
    
    #initialize lists for each measure
    comNDCG = []
    comRKL = []
    comDTR = []
    comDIR = []
    comMAP = []
    comAP = []
    

    for row in resultsForWN:
        #check if ndcg
        if row[1] == M_NDCG:
            #save data for comparison
            comNDCG.append(row)  
        #check if rKL
        elif row[1] == M_RKL:
            #save data for comparison
            comRKL.append(row)
        #check if DTR
        elif row[1] == M_DTR:
            #save data for comparison
            comDTR.append(row)
        #check if DIR
        elif row[1] == M_DIR:
            #save data for comparison
            comDIR.append(row)
        #check if MAP
        elif row[1] == M_MAP:
            #save data for comparison
            comMAP.append(row) 
        elif row[1] == M_AP:
            #save data for comparison
            comAP.append(row) 
    
    return comNDCG, comRKL, comDTR, comDIR, comMAP, comAP

def compareMeasures(comNDCG,comRKL,comDTR,comDIR,comMAP, algoList):
    """
    Call comparison methods for each measure
    
    @param comNDCG: List with results for NDCG
    @param comRKL: List with results for RKL
    @param comDTR: List with results for DTR
    @param comDIR: List with results for DIR
    @param comMAP: List with results for MAP
    @param algoList: List with all algorithms in the benchmarking
    
    return lists for each measure with its evaluation results from the corresponding methods
    """
    comNDCG=(compareNDCG(comNDCG, algoList))
    comRKL=(compareRKL(comRKL, algoList))
    comDTR=(compareDTR(comDTR, algoList))
    comDIR=(compareDIR(comDIR, algoList))
    comMAP=(compareMAP(comMAP, algoList))
    
    return comNDCG, comRKL, comDTR, comDIR, comMAP
    
def compareNDCG(compareList, algoList):
    
    """
    Compare NDCG to receive the Winning Number of all algorithms for NDCG
    Plus get the Ideal Winning Number(IWN) per Algorithm. Since we always
    evaluate all algorithms for all measures if they can be evaluated on a 
    data set, we assume that the IWN is the same for all measures
    
    @param compareList: List with all algorithms evaluated on a given data set
    for NDCG. List looks like [algoName, 'NDCG', valueForMap]
    @param algoList: List with all algorithms in the benchmarking
    
    returns a list with items as  
    [algoName, 'NDCG', winning number per algorithm for given data set, IWN]
    """
    
    results = []
    algoCount = 0
    idealWNCount = 0
    
    for algo in algoList:
        algoCount = 0
        idealWNCount = 0
        for row in compareList:
            if algo == row[0]:
                value = row[2]
                for row in compareList:
                    idealWNCount += 1
                    if value > row[2]:
                        algoCount += 1
        results.append([algo, M_NDCG, algoCount, idealWNCount])
        
    return results

def compareRKL(compareList,algoList):
    
    """
    Compare rKL to receive the Winning Number of all algorithms for rKL
    
    @param compareList: List with all algorithms evaluated on a given data set
    for rKL. List looks like [algoName, 'rKL', valueForMap]
    @param algoList: List with all algorithms in the benchmarking
    
    returns a list with items as  
    [algoName, 'rKL', winning number per algorithm for given data set]
    """
    
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
    
    """
    Compare DTR to receive the Winning Number of all algorithms for DTR
    
    @param compareList: List with all algorithms evaluated on a given data set
    for DTR. List looks like [algoName, 'DTR', valueForMap]
    @param algoList: List with all algorithms in the benchmarking
    
    returns a list with items as  
    [algoName, 'DTR', winning number per algorithm for given data set]
    """
    
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
    
    """
    Compare DIR to receive the Winning Number of all algorithms for DIR
    
    @param compareList: List with all algorithms evaluated on a given data set
    for DIR. List looks like [algoName, 'DIR', valueForMap]
    @param algoList: List with all algorithms in the benchmarking
    
    returns a list with items as  
    [algoName, 'DIR', winning number per algorithm for given data set]
    """
    
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
    
    """
    Compare MAP to receive the Winning Number of all algorithms for MAP
    
    @param compareList: List with all algorithms evaluated on a given data set
    for MAP. List looks like [algoName, 'MAP', valueForMap]
    @param algoList: List with all algorithms in the benchmarking
    
    returns a list with items as  
    [algoName, 'MAP', winning number per algorithm for given data set]
    """
    
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
    
    """
    Calculate distance from 1
    
    @param val: Value for DIR or DTR
    
    return the distance from 1
    """
    
    if val < 1:
        return 1 - val
    else:
        return val - 1                
                
                
"""   
results = [['GermanCreditAge_25', 'Color-Blind', 'AP', 1.0], ['GermanCreditAge_25', 'Color-Blind', 'NDCG', 1.0], ['GermanCreditAge_25', 'Color-Blind', 'rKL', 0.0019415156418714987], ['GermanCreditAge_25', 'Color-Blind', 'DTR', 0.9513075970964544], ['GermanCreditAge_25', 'Color-Blind', 'DIR', 0.9394417693807077], ['GermanCredit_Age25', 'FeldmanEtAl', 'AP', 1.0], ['GermanCredit_Age25', 'FeldmanEtAl', 'NDCG', 0.9940294681008911], ['GermanCredit_Age25', 'FeldmanEtAl', 'rKL', 0.0017166872359973236], ['GermanCredit_Age25', 'FeldmanEtAl', 'DTR', 0.8508150326312649], ['GermanCredit_Age25', 'FeldmanEtAl', 'DIR', 0.8109361787503478], ['GermanCredit_Age25', 'FAIR', 'AP', 0.00774059274059274], ['GermanCredit_Age25', 'FAIR', 'NDCG', 0.9898008868890052], ['GermanCredit_Age25', 'FAIR', 'rKL', 0.001685462935765634], ['GermanCredit_Age25', 'FAIR', 'DTR', 1.1557388518132612], ['GermanCredit_Age25', 'FAIR', 'DIR', 1.0940549790492724], ['GermanCredit_Age25', 'LFRanking', 'AP', 1.0], ['GermanCredit_Age25', 'LFRanking', 'NDCG', 1.0], ['GermanCredit_Age25', 'LFRanking', 'rKL', 0.0019415156418714987], ['GermanCredit_Age25', 'LFRanking', 'DTR', 0.9513075970964544], ['GermanCredit_Age25', 'LFRanking', 'DIR', 0.9394417693807077], ['GermanCredit_Age35', 'Color-Blind', 'AP', 1.0], ['GermanCredit_Age35', 'Color-Blind', 'NDCG', 1.0], ['GermanCredit_Age35', 'Color-Blind', 'rKL', 0.013221817354072955], ['GermanCredit_Age35', 'Color-Blind', 'DTR', 1.0254952326065843], ['GermanCredit_Age35', 'Color-Blind', 'DIR', 1.0294769735762885], ['GermanCredit_Age35', 'FeldmanEtAl', 'AP', 1.0], ['GermanCredit_Age35', 'FeldmanEtAl', 'NDCG', 0.9821805792350532], ['GermanCredit_Age35', 'FeldmanEtAl', 'rKL', 0.016581041467974518], ['GermanCredit_Age35', 'FeldmanEtAl', 'DTR', 0.7609134808537342], ['GermanCredit_Age35', 'FeldmanEtAl', 'DIR', 0.7061008645947737], ['GermanCredit_Age35', 'FAIR', 'AP', 0.0], ['GermanCredit_Age35', 'FAIR', 'NDCG', 0.9821805792350532], ['GermanCreditAge_35', 'FAIR', 'rKL', 0.00520983810847107], ['GermanCredit_Age35', 'FAIR', 'DTR', 1.1083207622873845], ['GermanCredit_Age35', 'FAIR', 'DIR', 1.0406462482475403], ['GermanCredit_Age35', 'LFRanking', 'AP', 1.0], ['GermanCredit_Age35', 'LFRanking', 'NDCG', 1.0], ['GermanCredit_Age35', 'LFRanking', 'rKL', 0.013221817354072955], ['GermanCredit_Age35', 'LFRanking', 'DTR', 1.0254952326065843], ['GermanCredit_Age35', 'LFRanking', 'DIR', 1.0294769735762885], ['GermanlCreditSex', 'Color-Blind', 'AP', 1.0], ['GermanlCreditSex', 'Color-Blind', 'NDCG', 1.0], ['GermanlCreditSex', 'Color-Blind', 'rKL', 0.0008560666111084664], ['GermanlCreditSex', 'Color-Blind', 'DTR', 1.0522788238170926], ['GermanlCreditSex', 'Color-Blind', 'DIR', 1.066408846804308], ['GermanlCreditSex', 'FeldmanEtAl', 'AP', 1.0], ['GermanlCreditSex', 'FeldmanEtAl', 'NDCG', 0.9985107454401517], ['GermanlCreditSex', 'FeldmanEtAl', 'rKL', 0.013239711180642978], ['GermanlCreditSex', 'FeldmanEtAl', 'DTR', 1.1105073352926267], ['GermanlCreditSex', 'FeldmanEtAl', 'DIR', 1.1413786061494602], ['GermanlCreditSex', 'FAIR', 'AP', 0.02660704156954157], ['GermanlCreditSex', 'FAIR', 'NDCG', 0.9985107454401517], ['GermanlCreditSex', 'FAIR', 'rKL', 0.0], ['GermanlCreditSex', 'FAIR', 'DTR', 0.9862620820134426], ['GermanlCreditSex', 'FAIR', 'DIR', 1.0091629255201566], ['GermanlCreditSex', 'LFRanking', 'AP', 1.0], ['GermanlCreditSex', 'LFRanking', 'NDCG', 1.0], ['GermanlCreditSex', 'LFRanking', 'rKL', 0.0008560666111084664], ['GermanlCreditSex', 'LFRanking', 'DTR', 1.0522788238170926], ['GermanlCreditSex', 'LFRanking', 'DIR', 1.066408846804308]]


calculateFinalEvaluation(results, ['GermanlCreditSex.csv','GermanCredit_Age25.csv','GermanCredit_Age35.csv'])

"""