# -*- coding: utf-8 -*-
"""
Created on Sat May 26 14:01:41 2018

@author: Laura

"""

from collections import Counter

#constants for algorithms
ALGO_COLORBLIND = 'Color-Blind'
ALGO_FAIR = 'FAIR'
ALGO_LFRANKING = 'LFRanking'
ALGO_FELDMAN = 'FeldmanEtAl'
ALGO_FOEIRDPC = 'FOEIR-DPC'
ALGO_FOEIRDTC = 'FOEIR-DTC'
ALGO_FOEIRDIC = 'FOEIR-DIC'
ALGO_LISTNET = 'ListNet'

#constants for measures
M_NDCG1 = 'NDCG@1'
M_NDCG5 = 'NDCG@5'
M_NDCG10 = 'NDCG@10'
M_RKL = 'rKL'
M_DTR = 'DTR'
M_DIR = 'DIR'
M_MAP = 'MAP'
M_AP = 'AP'
M_FAK = 'FairnessAtK'

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
    
    algoList = [ALGO_COLORBLIND,ALGO_FAIR,ALGO_LFRANKING,ALGO_FELDMAN, ALGO_FOEIRDPC, ALGO_FOEIRDTC, ALGO_FOEIRDIC, ALGO_LISTNET]
    
    dataSets = []
    querySets = []
    
    resultsForWN = []
    midResults = []
    finalResults = []
    queryAlgoList = []
    
    lNDCG1 = []
    lNDCG5 = []
    lNDCG10 = []
    lRKL = []
    lDTR = []
    lDIR = []
    lMAP = []
    lAP = []
    lFaK = []

    
    #extract data set names
    for name in fileNames:
        if '_' not in name:
            name = name.split('.')
            dataSets.append(name[0])
        else:
            name = name.split('.')
            querySets.append(name[0])
            name = name[0].split('_')
            dataSets.append(name[0])
    
    #get the number of data sets for nwn     
    dataSetDic = Counter(dataSets)
    
    #getting unique elements
    dataSets = set(dataSets)
    dataSets = list(dataSets)
    
    
    
    #average evaluations over queries for a data set
    for name in dataSets:
        helpResults = []
        for row in results:
            #if name is equal to row just append row for NWN evaluation
            if name == row[0]:
                midResults.append(row)
            #elif name is only in row append row for averaging over queries
            elif name in row[0]:
                helpResults.append(row[1:])
                queryAlgoList.append(row[1])
        #check if the current data set is a query set
        if dataSetDic[name] != 1:
            #get the number of occurences of the algorithm in the subset
            queryAlgoDic = Counter(queryAlgoList)
            hNDCG1,hNDCG5,hNDCG10, hRKL, hDTR, hDIR, hMAP, hAP, hFaK = getListForMeasureInDataSet(helpResults, algoList)
            lNDCG1 += hNDCG1
            lNDCG5 += hNDCG5
            lNDCG10 += hNDCG10
            lRKL += hRKL
            lDTR += hDTR
            lDIR += hDIR
            lAP += hAP
            lFaK += hFaK
            midResults += calculateAverage(lNDCG1,lNDCG5,lNDCG10, lRKL, lDTR, lDIR, lAP, hFaK, name, algoList, queryAlgoDic)
            
    
    lNDCG1 = []
    lNDCG5 = []
    lNDCG10 = []
    lRKL = []
    lDTR = []
    lDIR = []
    lMAP = []
    lAP = []
    lFaK = []
    
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
        hNDCG1,hNDCG5,hNDCG10, hRKL, hDTR, hDIR, hMAP, hAP, hFaK = getListForMeasureInDataSet(resultsForWN, algoList)
        hNDCG1,hNDCG5,hNDCG10, hRKL, hDTR, hDIR, hMAP, hFaK = compareMeasures(hNDCG1,hNDCG5,hNDCG10, hRKL, hDTR, hDIR, hMAP, hFaK, algoList)
        lNDCG1 += hNDCG1
        lNDCG5 += hNDCG5
        lNDCG10 += hNDCG10
        lRKL += hRKL
        lDTR += hDTR
        lDIR += hDIR
        lMAP += hMAP
        lFaK += hFaK
    
    finalResults += calculateNWN(lNDCG1,lNDCG5,lNDCG10, lRKL, lDTR, lDIR, lMAP, lFaK, algoList)
    
    return finalResults
            
        

def calculateAverage(lNDCG1,lNDCG5,lNDCG10, lRKL, lDTR, lDIR, lAP, lFaK, dataSetName, algoList, queryAlgoDic):
    
    """
    Calculates the average values over a set of queries for each measure and algorithm
    
    @param lNDCG: list with values for all algorithms over issued queries for NDCG 
    @param lRKL: list with values for all algorithms over issued queries for rKL
    @param lDTR: list with values for all algorithms over issued queries for DTR
    @param lDIR: list with values for all algorithms over issued queries for DIR
    @param lMAP: list with values for all algorithms over issued queries for AP
    @param lFaK: list with values for all algorithms over issued queries for FairnessAtK
    @param dataSetName: Name of data set
    @param algoList: List with all algorithms in the benchmarking
    @param queryAlgoDic: A dictionary with the algorithm name and the total occurences in
    the subset of the results. This is necessary since some evaluations might not be possible
    on every query in a data set and using this we can find out how many queries were evaluated
    
    return result list with average values for each measure and each algorithm
    """
    
    actualAlgoList = []
    results = []
    
    #check which algorithms were evaluated on this data set
    for algo in algoList:
        for row in lNDCG1:
            if algo == row[0]:
                #append actually included algorithms for this data set
                actualAlgoList.append(algo)
                
    #make actualAlgoList a list of unique items
    actualAlgoList = set(actualAlgoList)
    actualAlgoList = list(actualAlgoList)
    
    
    for algo in actualAlgoList:
        resNDCG1 = 0
        resNDCG5 = 0
        resNDCG10 = 0
        resRKL = 0
        resDTR = 0
        resDIR = 0
        resAP = 0
        resFaK = 0
        for nDCG1, nDCG5, nDCG10, rKL, dTR, dIR, aP, faK in zip(lNDCG1,lNDCG5,lNDCG10, lRKL, lDTR, lDIR, lAP, lFaK):
            if algo == nDCG1[0]:
                resNDCG1 += nDCG1[2]
            if algo == nDCG5[0]:
                resNDCG5 += nDCG5[2]
            if algo == nDCG10[0]:
                resNDCG10 += nDCG10[2]    
            if algo == rKL[0]:
                resRKL += rKL[2]
            if algo == dTR[0]:
                resDTR += dTR[2]
            if algo == dIR[0]:
                resDIR += dIR[2]
            if algo == aP[0]:
                resAP += aP[2]
            if algo == aP[0]:
                resFaK += faK[2]
        #make sure count for data set is not 0 to prevent division by 0
        #results need to be averaged with the number of queries in the data set
        #since we count the occurence of each algorithm in the results we need to 
        #devide by the number of measures present in the evaluation
        if queryAlgoDic[algo] != 0:
            results.append([dataSetName, algo, M_NDCG1, resNDCG1/(queryAlgoDic[algo]/9)])
            results.append([dataSetName, algo, M_NDCG5, resNDCG5/(queryAlgoDic[algo]/9)])
            results.append([dataSetName, algo, M_NDCG10, resNDCG10/(queryAlgoDic[algo]/9)])
            results.append([dataSetName, algo, M_MAP, resAP/(queryAlgoDic[algo]/9)])
            results.append([dataSetName, algo, M_RKL, resRKL/(queryAlgoDic[algo]/9)]) 
            results.append([dataSetName, algo, M_DTR, resDTR/(queryAlgoDic[algo]/9)]) 
            results.append([dataSetName, algo, M_DIR, resDIR/(queryAlgoDic[algo]/9)]) 
            results.append([dataSetName, algo, M_FAK, resFaK/(queryAlgoDic[algo]/9)]) 
        
        
    return results
    
  
def calculateNWN(lNDCG1,lNDCG5,lNDCG10, lRKL, lDTR, lDIR, lMAP, lFaK, algoList):
        
    """
    @param lNDCG: list with WN for all algorithms for NDCG plus the IWN
    @param lRKL: list with WN for all algorithms for rKL
    @param lDTR: list with WN for all algorithms for DTR
    @param lDIR: list with WN for all algorithms for DIR
    @param lMAP: list with WN for all algorithms for MAP
    @param lFaK: list with WN for all algorithms for FairnessAtK
    @param algoList: List with all algorithms in the benchmarking
    
    returns list with NWN for each algorithm on each measure
    """
    
    idealWinningNumber = 0
    
    resultsNWN = []
    
    for algo in algoList:
        resNDCG1 = 0
        resNDCG5 = 0
        resNDCG10 = 0
        resRKL = 0
        resDTR = 0
        resDIR = 0
        resMAP = 0
        resFaK = 0
        idealWinningNumber = 0
        for nDCG1,nDCG5,nDCG10, rKL, dTR, dIR, mAP, faK in zip(lNDCG1,lNDCG5,lNDCG10, lRKL, lDTR, lDIR, lMAP, lFaK):
            if algo == nDCG1[0]:
                resNDCG1 += nDCG1[2]
                idealWinningNumber += nDCG1[3]
            if algo == nDCG5[0]:
                resNDCG5 += nDCG5[2]
            if algo == nDCG10[0]:
                resNDCG10 += nDCG10[2] 
            if algo == rKL[0]:
                resRKL += rKL[2]
            if algo == dTR[0]:
                resDTR += dTR[2]
            if algo == dIR[0]:
                resDIR += dIR[2]
            if algo == mAP[0]:
                resMAP += mAP[2]
            if algo == mAP[0]:
                resFaK += faK[2]
        #make sure IWN is not 0

        if idealWinningNumber != 0:
            resultsNWN.append(['NWN', algo, M_NDCG1, resNDCG1/idealWinningNumber])
            resultsNWN.append(['NWN', algo, M_NDCG5, resNDCG5/idealWinningNumber])
            resultsNWN.append(['NWN', algo, M_NDCG10, resNDCG10/idealWinningNumber])
            resultsNWN.append(['NWN', algo, M_MAP, resMAP/idealWinningNumber])
            resultsNWN.append(['NWN', algo, M_RKL, resRKL/idealWinningNumber])
            resultsNWN.append(['NWN', algo, M_DTR, resDTR/idealWinningNumber])
            resultsNWN.append(['NWN', algo, M_DIR, resDIR/idealWinningNumber])
            resultsNWN.append(['NWN', algo, M_FAK, resFaK/idealWinningNumber])

        
    return resultsNWN
       
def getListForMeasureInDataSet(resultsForWN, algoList):
    
    """
    Gets result lists for each measure in a data set
    
    @param resultsForWN: list with results from the evaluation measures on one
    data set
    @param algoList: List with all algorithms in the benchmarking
    
    return lists for each measure with its evaluation results from the corresponding methods
    """
    
    #initialize lists for each measure
    comNDCG1 = []
    comNDCG5 = []
    comNDCG10 = []
    comRKL = []
    comDTR = []
    comDIR = []
    comMAP = []
    comAP = []
    comFaK = []
    

    for row in resultsForWN:
        #check if ndcg
        if row[1] == M_NDCG1:
            #save data for comparison
            comNDCG1.append(row) 
        elif row[1] == M_NDCG5:
            #save data for comparison
            comNDCG5.append(row)  
        elif row[1] == M_NDCG10:
            #save data for comparison
            comNDCG10.append(row)  
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
        elif row[1] == M_FAK:
            #save data for comparison
            comFaK.append(row) 
    
    return comNDCG1,comNDCG5,comNDCG10, comRKL, comDTR, comDIR, comMAP, comAP, comFaK

def compareMeasures(comNDCG1,comNDCG5,comNDCG10,comRKL,comDTR,comDIR,comMAP, comFaK, algoList):
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
    comNDCG1=(compareGreaterThan(comNDCG1, algoList, M_NDCG1))
    comNDCG5=(compareGreaterThan(comNDCG5, algoList, M_NDCG5))
    comNDCG10=(compareGreaterThan(comNDCG10, algoList, M_NDCG10))
    comRKL=(compareSmallerThan(comRKL, algoList, M_RKL))
    comDTR=(compareDist(comDTR, algoList, M_DTR))
    comDIR=(compareDist(comDIR, algoList, M_DIR))
    comMAP=(compareGreaterThan(comMAP, algoList, M_MAP))
    comFaK=(compareGreaterThan(comFaK,algoList,M_FAK))
    
    return comNDCG1,comNDCG5,comNDCG10, comRKL, comDTR, comDIR, comMAP, comFaK
    
def compareGreaterThan(compareList, algoList, measureName):
    
    """
    Compare comparison greater than to receive the Winning Number of all 
    algorithms for NDCG@1, NDCG@5, NDCG@10, and MAP
    Plus get the Ideal Winning Number(IWN) per Algorithm. Since we always
    evaluate all algorithms for all measures if they can be evaluated on a 
    data set, we assume that the IWN is the same for the other measures (rKL,
    DIR, DTR) as well
    
    @param compareList: List with all algorithms evaluated on a given data set
    for a given measure. List looks like [algoName, measureName, valueForMeasure]
    @param algoList: List with all algorithms in the benchmarking
    @param measureName: Name of the evaluated measure
    
    returns a list with items as  
    [algoName, measureName, winning number per algorithm for given data set, IWN]
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
        results.append([algo, measureName, algoCount, idealWNCount])
        
    return results

def compareSmallerThan(compareList,algoList,measureName,):
    
    """
    Compares to receive the Winning Number of all algorithms for a smaller than
    relation such as needed for rKL
    
    @param compareList: List with all algorithms evaluated on a given data set
    for rKL. List looks like [algoName, 'rKL', valueForMap]
    @param algoList: List with all algorithms in the benchmarking
    @param measureName: Name of the evaluated measure
    
    returns a list with items as  
    [algoName, measureName, winning number per algorithm for given data set]
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
        results.append([algo, measureName, algoCount])
        
    return results

def compareDist(compareList, algoList, measureName,):
    
    """
    Compare DTR to receive the Winning Number of all algorithms for DTR
    
    @param compareList: List with all algorithms evaluated on a given data set
    for DTR. List looks like [algoName, 'DTR', valueForMap]
    @param algoList: List with all algorithms in the benchmarking
    @param measureName: Name of the evaluated measure
    
    returns a list with items as  
    [algoName, measureName winning number per algorithm for given data set]
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
        results.append([algo, measureName, algoCount])
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
                
                
