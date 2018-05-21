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
#from candidateCreator.candidate import Candidate

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
    
    #initialize list for evaluation results
    evalResults = []
    
    #creates a CSV file in preprocessedDataSets as described in csvPreprocessing
    nameOutCredit25 = cP.createScoreOrderedCSV("dataSets/GermanCredit/GermanCredit_age25.csv", 2)
    nameOutCredit35 = cP.createScoreOrderedCSV("dataSets/GermanCredit/GermanCredit_age35.csv", 2)
    
    #creates Candidates from the preprocessed CSV files in folder preprocessedDataSets
    protectedCredit25, nonProtectedCredit25, rankingCredit25 = cC.create(nameOutCredit25)
    protectedCredit35, nonProtectedCredit35, rankingCredit35 = cC.create(nameOutCredit35)
    
    createRankingCSV(rankingCredit25, 'ColorBlind/GermanCredit_age25preranking.csv' )
    createRankingCSV(rankingCredit35, 'ColorBlind/GermanCredit_age35preranking.csv' )
    
    lenCredit25 = len(rankingCredit25)
    lenCredit35 = len(rankingCredit35)
    
    #extract Data set name from path, will have 'pre' for preprocessed at the end
    creditNameAge25 = extractDataSetName(nameOutCredit25)
    creditNameAge35 = extractDataSetName(nameOutCredit35)

    evalResults.append(runMetrics(100, lenCredit25, protectedCredit25, nonProtectedCredit25, rankingCredit25, creditNameAge25, 'Color-Blind'))
    evalResults.append(runMetrics(100, lenCredit35, protectedCredit35, nonProtectedCredit35, rankingCredit35, creditNameAge35, 'Color-Blind'))
    
    #run evaluations for FAIR
    #run FAIR algorithm on all available data sets
    FAIRRankingOutCreditAge25, notSelectedCreditAge25, pathFAIRCreditAge25 = runFAIR(creditNameAge25, protectedCredit25, nonProtectedCredit25, 100)
    FAIRRankingOutCreditAge35, notSelectedCreditAge35, pathFAIRCreditAge35 = runFAIR(creditNameAge35, protectedCredit35, nonProtectedCredit35, 100)
    #Update the currentIndex of a candidate according to FAIR
    FAIRRankingOutCreditAge25 = updateCurrentIndex(FAIRRankingOutCreditAge25)
    FAIRRankingOutCreditAge35 = updateCurrentIndex(FAIRRankingOutCreditAge35)
    #create CSV with rankings from FAIR
    createRankingCSV(FAIRRankingOutCreditAge25, pathFAIRCreditAge25)
    createRankingCSV(FAIRRankingOutCreditAge35, pathFAIRCreditAge35)
    #evaluate FAIR with all available measures
    evalResults.append(runMetrics(100, lenCredit25, protectedCredit25, nonProtectedCredit25, FAIRRankingOutCreditAge25, creditNameAge25, 'FAIR'))
    evalResults.append(runMetrics(100, lenCredit35, protectedCredit35, nonProtectedCredit35, FAIRRankingOutCreditAge35, creditNameAge35, 'FAIR'))
    
    #run evaluations for LFRanking
    #run LFRanking algorithm on all available data sets
    LFRankingOutCreditAge25, pathLFRankingCreditAge25 = runLFRanking(rankingCredit25,protectedCredit25,nonProtectedCredit25,4,creditNameAge25)
    LFRankingOutCreditAge35, pathLFRankingCreditAge35 = runLFRanking(rankingCredit35,protectedCredit35,nonProtectedCredit35,4,creditNameAge35)    
    #create CSV file with ranking outputs
    createRankingCSV(LFRankingOutCreditAge25, pathLFRankingCreditAge25)
    createRankingCSV(LFRankingOutCreditAge35, pathLFRankingCreditAge35)
    #Update the currentIndex of a candidate according to LFRanking
    LFRankingOutCreditAge25 = updateCurrentIndex(LFRankingOutCreditAge25)
    LFRankingOutCreditAge35 = updateCurrentIndex(LFRankingOutCreditAge35)
    #evaluate LFRanking with all available measures
    evalResults.append(runMetrics(100, lenCredit25, protectedCredit25, nonProtectedCredit25, LFRankingOutCreditAge25, creditNameAge25, 'LFRanking'))
    evalResults.append(runMetrics(100, lenCredit35, protectedCredit35, nonProtectedCredit35, LFRankingOutCreditAge35, creditNameAge35, 'LFRanking'))
    
    print (evalResults)
    
    
def extractDataSetName(filePath):
    """
    @param filePath: file path to extract the name of the data set from
    
    return the file name from the given path 
    """
    helper = filePath.split("/")
    dataSetNameAndFormat = helper[-1].split(".")
    dataSetName = dataSetNameAndFormat[0]
    
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