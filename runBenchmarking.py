# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:43:30 2018

@author: Laura
"""

from csvProcessing.csvPreprocessing import csvPreprocessing as cP
from candidateCreator.createCandidate import createCandidate as cC
from csvProcessing.csvPrintRanking import createRankingCSV
from algorithms.fair_ranker.runRankFAIR import runFAIR
from algorithms.LFRanking.runLFRanking import runLFRanking
from measures.runMetrics import runMetrics
import measures.relevance as rel

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
    
    #creates a CSV file in preprocessedDataSets as described in csvPreprocessing
    filePathCredit25 = cP.createScoreOrderedCSV("dataSets/GermanCredit/GermanCredit_age25.csv", 2)
    results += (benchmarkingProcess(filePathCredit25, 100))
    filePathCredit35 = cP.createScoreOrderedCSV("dataSets/GermanCredit/GermanCredit_age35.csv", 2)
    results += (benchmarkingProcess(filePathCredit35, 100))
    filePathCreditSex = cP.createScoreOrderedCSV("dataSets/GermanCredit/GermanCredit_sex.csv", 2)
    results += (benchmarkingProcess(filePathCreditSex, 100))
    
    results += (rel.calculateMAP('GermanCredit', results))
    
    print(results)
    
    
def benchmarkingProcess(dataSetPath, k):
    
    #initialize list for evaluation results
    evalResults = []
    
    #creates Candidates from the preprocessed CSV files in folder preprocessedDataSets
    protected, nonProtected, originalRanking = cC.create(dataSetPath)
    
    #extract Data set name from path
    dataSetName = extractDataSetName(dataSetPath)
    
    #creates a csv with candidates ranked with color-blind ranking
    createRankingCSV(originalRanking, 'ColorBlind/' + dataSetName + 'ranking.csv' )

    #run the metrics ones for the color-blind ranking
    evalResults += (runMetrics(k, protected, nonProtected, originalRanking, originalRanking, dataSetName, 'Color-Blind'))
    
    #run evaluations for FAIR
    #run FAIR algorithm 
    FAIRRanking, notSelected, pathFAIR = runFAIR(dataSetName, protected, nonProtected, k)
    #Update the currentIndex of a candidate according to FAIR
    FAIRRanking = updateCurrentIndex(FAIRRanking)
    #create CSV with rankings from FAIR
    createRankingCSV(FAIRRanking, pathFAIR)
    #evaluate FAIR with all available measures
    evalResults += (runMetrics(k, protected, nonProtected, FAIRRanking, originalRanking, dataSetName, 'FAIR'))
        
    #run evaluations for LFRanking
    #run LFRanking algorithm
    LFRanking, pathLFRanking = runLFRanking(originalRanking, protected, nonProtected, 4, dataSetName) 
    #create CSV file with ranking outputs
    createRankingCSV(LFRanking, pathLFRanking)
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