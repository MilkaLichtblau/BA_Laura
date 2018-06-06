# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:43:30 2018

@author: Laura
"""

from src.csvProcessing.csvPreprocessing import csvPreprocessing as cP
from src.candidateCreator.createCandidate import createCandidate as cC
from src.csvProcessing.csvPrinting import createRankingCSV
from src.algorithms.fair_ranker.runRankFAIR import runFAIR
from src.algorithms.LFRanking.runLFRanking import runLFRanking
from src.algorithms.FeldmanEtAl.runFeldmanEtAl import feldmanRanking
from src.algorithms.FOEIR.runFOEIR import runFOEIR
from src.measures.runMetrics import runMetrics
from src.visualizer.visualizeData import plotNWN
import src.measures.finalEvaluation as finalEval
import os
import pandas as pd
import numpy as np

"""

This method runs the whole benchmarking process.
We currently create rankings using every algorithm on each data set available in the benchmarking
and generate evaluations with the help of the implemented measures afterwards
Since we create candidates objects only once for each data set, evaluations
should always be done on the originalQualification attribute and then saved in 
the qualification attribute. Furthermore, all evaluations on the same data set
(same set of candidates) need to be done directly subsequently, i.e.
each algorithm should be evaluated completely for one data set (run, CSV creation,
and measure evalution) before the candidate list is passed on to the next 
algorithm.

IMPORTANT: An underscore in a file name denotes a query in our framework, hence
the file will be treated and evaluated as if it belonged to one data set
"""

def main():
    
    """
    This method starts the whole benchmkaring process. It first reads all 
    raw data sets and creates CSV files from those.
    Then it calls the method benchmarkingProcess to run the benchmarks on
    the implemented algorithms and evalute those with provided measures.
    The method then
    
    """
    #initialize list for evaluation results
    results = []
    finalResults = []
    fileNames = []
    
    """
    paths = []

    for dirpath, dirnames, files in os.walk('scoredDataSets/'):
        if dirnames != []:
            paths += dirnames
            print(paths)

    for folder in paths:
        for dirpath, dirname, files in os.walk('scoredDataSets/' + folder + '/'):
            print(files)
                if folder == 'GermanCredit':
                    filePathCredit25 = cP.createScoreOrderedCSV("scoredDataSets/GermanCredit/" + files, 2)
                    results += (scoreBasedEval(filePathCredit25, 100))
    """
    """
    for dirpath, dirnames, files in os.walk("learningDataSets//"):
        for name in files:
            if 'csv' in name:
                fileNames.append(name)
                filePathCredit25 = cP.createScoreOrderedCSV("scoredDataSets/GermanCredit/" + name, 2)
                results += (scoreBasedEval(filePathCredit25, 100))
    """
    
    for dirpath, dirnames, files in os.walk("scoredDataSets/GermanCredit/"):
        for name in files:
            if 'csv' in name:
                fileNames.append(name)
                filePath = cP.createScoreOrderedCSV("scoredDataSets/GermanCredit/" + name, 2)
                results += (scoreBasedEval(filePath, 100))
        
    for dirpath, dirnames, files in os.walk("scoredDataSets/COMPAS/"):
        for name in files:
            if 'csv' in name:
                fileNames.append(name)
                filePath = cP.createScoreOrderedCSV("scoredDataSets/COMPAS/" + name, 2)
                results += (scoreBasedEval(filePath, 100))        
    
    
    finalResults = finalEval.calculateFinalEvaluation(results, fileNames)  

    #print (finalResults)          
    
    df = pd.DataFrame(np.array(finalResults).reshape(len(finalResults),4), columns = ['Data_Set_Name', 'Algorithm_Name', 'Measure', 'Value'])
    
    df.to_csv('results/evaluationResults.csv', index=(False))
    
    plotNWN()
    
def scoreBasedEval(dataSetPath, k):
    
    """
    Starts the optimization and evaluation of the post-processing methods
    
    @param dataSetPath: Path of the data sets storing scores in the first column and
    membership of the sensitive group in the second column
    @param k: Provides the length of the ranking
    
    returns a list of evaluation results of the form:
        [dataSetName, Optimization Algorithm, Measure, Value of Measure]
    """
    
    #initialize list for evaluation results
    evalResults = []
    
    #creates Candidates from the preprocessed CSV files in folder preprocessedDataSets
    protected, nonProtected, originalRanking = cC.createScoreBased(dataSetPath)
    
    #extract Data set name from path
    dataSetName = extractDataSetName(dataSetPath)
    
    #creates a csv with candidates ranked with color-blind ranking
    createRankingCSV(originalRanking, 'Color-Blind/' + dataSetName + 'ranking.csv',k )
    #run the metrics ones for the color-blind ranking
    evalResults += (runMetrics(k, protected, nonProtected, originalRanking, originalRanking, dataSetName, 'Color-Blind'))
    
    
    #create ranking like Feldman et al.
    feldRanking, pathFeldman = feldmanRanking(protected, nonProtected, k, dataSetName)
    #create CSV with rankings from FAIR
    createRankingCSV(feldRanking, pathFeldman,k)
    #evaluate FAIR with all available measures
    evalResults += (runMetrics(k, protected, nonProtected, feldRanking, originalRanking, dataSetName, 'FeldmanEtAl'))
    
    
    #run evaluations for FOEIR with different Fairness Constraints
    #run for FOEIR-DPC
    dpcRanking, dpcPath, isDPC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DPC', 40)
    if isDPC == True:
        evalResults += (runMetrics(40, protected, nonProtected, originalRanking, originalRanking, dataSetName, 'FOEIR-DPC'))
        createRankingCSV(dpcRanking, dpcPath,40)
        
    dtcRanking, dtcPath, isDTC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DTC', 40)
    if isDTC == True:
        evalResults += (runMetrics(40, protected, nonProtected, originalRanking, originalRanking, dataSetName, 'FOEIR-DTC'))
        createRankingCSV(dtcRanking, dtcPath,40)
        
    dicRanking, dicPath, isDIC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DIC', 40)
    if isDIC == True:
        createRankingCSV(dicRanking, dicPath,40)
        evalResults += (runMetrics(40, protected, nonProtected, originalRanking, originalRanking, dataSetName, 'FOEIR-DIC'))
            
    #run evaluations for FAIR
    #run FAIR algorithm 
    FAIRRanking, notSelected, pathFAIR = runFAIR(dataSetName, protected, nonProtected, k)
    #Update the currentIndex of a candidate according to FAIR
    FAIRRanking = updateCurrentIndex(FAIRRanking)
    #create CSV with rankings from FAIR
    createRankingCSV(FAIRRanking, pathFAIR,k)
    #evaluate FAIR with all available measures
    evalResults += (runMetrics(k, protected, nonProtected, FAIRRanking, originalRanking, dataSetName, 'FAIR'))
        
    #run evaluations for LFRanking
    #run LFRanking algorithm
    LFRanking, pathLFRanking = runLFRanking(originalRanking, protected, nonProtected, 4, dataSetName) 
    #create CSV file with ranking outputs
    createRankingCSV(LFRanking, pathLFRanking,k)
    #Update the currentIndex of a candidate according to LFRanking
    LFRanking = updateCurrentIndex(LFRanking)
    #evaluate LFRanking with all available measures
    evalResults += (runMetrics(k, protected, nonProtected, LFRanking, originalRanking, dataSetName, 'LFRanking'))
    
    return evalResults
    
    
def extractDataSetName(filePath):
    """
    Extract the name of a data set from a file path
    @param filePath: file path to extract the name of the data set from
    
    return the file name from the given path 
    """
    helper = filePath.split("/")
    dataSetNameAndFormat = helper[-1].split(".")
    dataSetName = dataSetNameAndFormat[0]
    dataSetName = str.replace(dataSetName, 'pre','')
    
    return dataSetName



def updateCurrentIndex(ranking):
    
    """
    Updates the currentIndex of a ranking according to the current order in the
    list
    @param ranking: list with candidates of a ranking
    
    return list of candidates with updated currentIndex according to their 
    position in the current ranking
    
    """
    
    index = 0
    
    for i in range(len(ranking)):
        index += 1
        
        ranking[i].currentIndex = index
        
    return ranking

    
main()