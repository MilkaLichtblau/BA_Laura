# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:43:30 2018

@author: Laura
"""

from csvProcessing.csvPreprocessing import csvPreprocessing as cP
from candidateCreator.createCandidate import createCandidate as cC
from csvProcessing.csvPrintRanking import createRankingCSV
from algorithms.fair_ranker.runRankFAIR import run
#from candidateCreator.candidate import Candidate

"""

This method runs the whole benchmarking process.
We currently create rankings using every algorithm on each data set available in the benchmarking
and generate evaluations with the help of the implemented measures afterwards

"""

def main():
    
    #creates a CSV file in preprocessedDataSets as described in csvPreprocessing
    nameOutCredit25, orderCredit25 = cP.createScoreOrderedCSV("dataSets/GermanCredit/GermanCredit_age25.csv", 2, desc=True)
    
    #creates Candidates from the preprocessed CSV files in folder preprocessedDataSets
    protectedCredit25, nonProtectedCredit25, rankingCredit25 = cC.create(nameOutCredit25, orderCredit25)
    
    dataSetName = extractDataSetName(nameOutCredit25)
    
    fairRankingOut, notSelected, rankingResultsPath = run(dataSetName, protectedCredit25, nonProtectedCredit25, 100)
    
    createRankingCSV(fairRankingOut, rankingResultsPath)
    
    
def extractDataSetName(filePath):
    
    helper = filePath.split("/")
    dataSetNameAndFormat = helper[-1].split(".")
    dataSetName = dataSetNameAndFormat[0]
    
    return dataSetName

    
main()