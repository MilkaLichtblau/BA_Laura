# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:43:30 2018

@author: Laura
"""


from src.candidateCreator.createCandidate import createCandidate as cC
from src.csvProcessing.csvPrinting import createRankingCSV
from src.algorithms.fair_ranker.runRankFAIR import runFAIR
from src.algorithms.LFRanking.runLFRanking import runLFRanking
from src.algorithms.FeldmanEtAl.runFeldmanEtAl import feldmanRanking
from src.algorithms.FOEIR.runFOEIR import runFOEIR
from src.algorithms.ListNet.runListNet import runListNet
from src.measures.runMetrics import runMetrics
from src.visualizer.visualizeData import plotData
import src.measures.finalEvaluation as finalEval
import os
import pandas as pd
import numpy as np
import csv
import datetime

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
    
    startTime = datetime.datetime.now()
    
    #read all data sets in TREC including all folds
    for dirpath, dirnames, files in os.walk("learningDataSets/TREC/"):
        
        if 'fold' in dirpath:
            
            #construct extractions for different folds
            getTrain = dirpath+'/train.csv'
            getValidation = dirpath+'/validation.csv'
            getTest = dirpath+'/test.csv'
            #constructs a candidate list for the test data set
            ranking, queryNumbers = cC.createLearningCandidate(getTest)
            
            #run ListNet learning process
            listNetRanking, dataSetName  = runListNet(ranking, getTrain, getValidation, getTest)
            #evaluate listNet learning process, print ranked queries and start scoreBasedEval
            listResults, listFileNames = evaluateLearning('ListNet',listNetRanking, dataSetName, queryNumbers, True)
            results += listResults
            fileNames += listFileNames
    
    
    #read all data sets in German Credit
    for dirpath, dirnames, files in os.walk("scoredDataSets/GermanCredit/"):
        for name in files:
            if 'csv' in name:
                dataSetName = getDataSetName(name)
                fileNames.append(dataSetName)
                results += (scoreBasedEval(dataSetName, "scoredDataSets/GermanCredit/" + name, 100))
    
    #read all data sets in COMPAS
    for dirpath, dirnames, files in os.walk("scoredDataSets/COMPAS/"):
        for name in files:
            if 'csv' in name:
                dataSetName = getDataSetName(name)
                fileNames.append(dataSetName)
                results += (scoreBasedEval(dataSetName, "scoredDataSets/COMPAS/" + name, 100, False))        
    
    finalResults = finalEval.calculateFinalEvaluation(results, fileNames)          
    
    df = pd.DataFrame(np.array(finalResults).reshape(len(finalResults),4), columns = ['Data_Set_Name', 'Algorithm_Name', 'Measure', 'Value'])
    
    df.to_csv('results/evaluationResults.csv', index=(False))
     
    plotData()
   
    endTime = datetime.datetime.now()
    
    print("Total time of execution: "+str(endTime-startTime))
    
    
def evaluateLearning(algoName, ranking, dataSetName, queryNumbers, listNet = False, k = 100):
    """
    Evaluates the learning algorithms per query, creates an output file for each ranked query,
    and start the scoreBasedEval method for each query
    
    @param algoName: Name of the algorithm which created the query rankings
    @param ranking: A list of candidates from different queries with new calculated scores for them
    @param dataSetName: Name of the data set without query numbers
    @param queryNumbers: List of query identifiers
    @param k: turncation point of the ranking
    
    return evalResults list with the evaluation results for the algorithms
           evalResults looks like this: [dataSetName, Optimization Algorithm, Measure, Value of Measure]
           fileNames list with file names for each query.
    """
    #initialize list for evaluation results
    evalResults = []
    fileNames = []
    
    #initialize k for evaluation purposes. This k is also used for calculation of FOIER algorithms
    evalK = k
    
    #check if evalK is not larger than 40
    if evalK > 40:
        print('Evaluations only done for k = 40 due to comparability reasons. Rankings are still created for '+str(k)+'. If changes to this are wished, please open runBenchmarking and change line 226 accordingly.')
        evalK = 40
    
    #loop for each query
    for query in queryNumbers:
    
        queryRanking = []
        queryProtected = []
        queryNonprotected = []
        output = []
        finalPrinting = [['Original_Score','learned_Scores','Ranking_Score_from_Postprocessing','Sensitive_Attribute']]
        #loop over the candidate list to construct the output
        for i in range(len(ranking)):
            
            #check if the query numbers are equal
            if ranking[i].query == query:
                
                originQ = str(ranking[i].originalQualification)
                learned = str(ranking[i].learnedScores)
                quali = str(ranking[i].qualification)
                proAttr = str(ranking[i].isProtected)
                
                output.append([originQ, learned, quali, proAttr])         
        
                #construct list with candiates for one query
                queryRanking.append(ranking[i])
                    
                if proAttr == 'True':
                    queryProtected.append(ranking[i])
                else:
                    queryNonprotected.append(ranking[i])
            
        finalName = dataSetName +'_'+str(query)
        
        fileNames.append(finalName)
        
        # sort candidates by credit scores 
        queryProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)
        queryNonprotected.sort(key=lambda candidate: candidate.qualification, reverse=True)
            
        #sorting the ranking in accordance with is new scores
        queryRanking.sort(key=lambda candidate: candidate.qualification, reverse=True)
        
        #update index accoring to the ranking
        queryRanking = updateCurrentIndex(queryRanking)
        queryRanking = updateLearnedIndex(queryRanking)
        #evaluate listNet
        evalResults += (runMetrics(evalK, queryProtected, queryNonprotected, queryRanking, queryRanking, finalName, 'ListNet'))
            
        output.sort(key=lambda x: x[2], reverse=True)
        
        finalPrinting += output
        
        #only start scoreBasedEval if the algorithm is listNet (baseline)
        if listNet == True:
            #run the score based evaluation on the ranked candidate list
            evalResults += scoreBasedEval(finalName,"", k, True, queryProtected, queryNonprotected, queryRanking, listNet)
            
        try:     
            with open('rankings/'+algoName+'/' + finalName +'ranking.csv','w',newline='') as mf:
                writer = csv.writer(mf)
                writer.writerows(finalPrinting) 
        except Exception:
            raise Exception("Some error occured during file creation. Double check specifics.")
            pass
    
    return evalResults, fileNames
    
def scoreBasedEval(dataSetName, dataSetPath, k = 100, features = True, protected = [], nonProtected = [], originalRanking = [], listNet = False):
    
    """
    Evaluates the learning to rank algorithms and runs 
    the optimization and evaluation of the post-processing methods
    
    @param dataSetName: Name of the data set
    @param dataSetPath: Path of the data sets for score based evaluation. 
    @param k: Provides the length of the ranking
    @param features: True if the provided data set has features for LFRanking, otherwise
    false
    @param protected: If data comes from a learning to rank algorithm this param holds a 
    list of candidates with protected group membership
    @param protected: If data comes from a learning to rank algorithm this param holds a 
    list of candidates with non-protected group membership
    @param protected: If data comes from a learning to rank algorithm this param holds a 
    list of candidates from the new ranking
    @param scoreData: Is set false if the data does not come from an already scored data 
    set but from a learning to rank algorithm
    
    returns a list of evaluation results of the form:
        [dataSetName, Optimization Algorithm, Measure, Value of Measure]
    """
    
    evalResults = []
    
    #initialize k for evaluation purposes. This k is also used for calculation of FOIER algorithms
    evalK = k
    
    #check if evalK is not larger than 40
    if evalK > 40:
        print('Evaluations only done for k = 40 due to comparability reasons. Rankings are still created for '+str(k)+'. If changes to this are wished, please open runBenchmarking and change line 226 accordingly.')
        evalK = 40
    
    #check if the given data comes from the base line algorithm ListNet
    #if it does not, construct candidates from the data
    if listNet == False:
        #creates Candidates from the preprocessed CSV files in folder preprocessedDataSets
        protected, nonProtected, originalRanking = cC.createScoreBased(dataSetPath)
    
    #creates a csv with candidates ranked with color-blind ranking
    createRankingCSV(originalRanking, 'Color-Blind/' + dataSetName + 'ranking.csv',k )
    #run the metrics ones for the color-blind ranking
    evalResults += (runMetrics(evalK, protected, nonProtected, originalRanking, originalRanking, dataSetName, 'Color-Blind'))
    
    
    #create ranking like Feldman et al.
    feldRanking, pathFeldman = feldmanRanking(protected, nonProtected, k, dataSetName)
    #Update the currentIndex of a candidate according to feldmanRanking
    feldRanking = updateCurrentIndex(feldRanking)
    #create CSV with rankings from FAIR
    createRankingCSV(feldRanking, pathFeldman,k)
    #evaluate FAIR with all available measures
    evalResults += (runMetrics(evalK, protected, nonProtected, feldRanking, originalRanking, dataSetName, 'FeldmanEtAl'))
    
    
    #run evaluations for FOEIR with different Fairness Constraints
    #we only produce rankings of k = 50 since construction of P as well as dicomposition of Birkhoff take a very long time
    #and consume a lot of memory.
    #run for FOEIR-DPC
    dpcRanking, dpcPath, isDPC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DPC', evalK)
    if isDPC == True:
        dpcRanking = updateCurrentIndex(dpcRanking)
        createRankingCSV(dpcRanking, dpcPath,evalK)
        evalResults += (runMetrics(evalK, protected, nonProtected, dpcRanking, originalRanking, dataSetName, 'FOEIR-DPC'))
        
    dtcRanking, dtcPath, isDTC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DTC', evalK)
    if isDTC == True:
        dtcRanking = updateCurrentIndex(dtcRanking)
        createRankingCSV(dtcRanking, dtcPath,evalK)
        evalResults += (runMetrics(evalK, protected, nonProtected, dtcRanking, originalRanking, dataSetName, 'FOEIR-DTC'))
        
    dicRanking, dicPath, isDIC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DIC', evalK)
    if isDIC == True:
        dicRanking = updateCurrentIndex(dicRanking)
        createRankingCSV(dicRanking, dicPath,evalK)
        evalResults += (runMetrics(evalK, protected, nonProtected, dicRanking, originalRanking, dataSetName, 'FOEIR-DIC'))
          
    #run evaluations for FAIR
    #run FAIR algorithm 
    FAIRRanking, notSelected, pathFAIR = runFAIR(dataSetName, protected, nonProtected, k)
    #Update the currentIndex of a candidate according to FAIR
    FAIRRanking = updateCurrentIndex(FAIRRanking)
    #create CSV with rankings from FAIR
    createRankingCSV(FAIRRanking, pathFAIR,k)
    #evaluate FAIR with all available measures
    evalResults += (runMetrics(evalK, protected, nonProtected, FAIRRanking, originalRanking, dataSetName, 'FAIR'))
    
    if features:
        try:
            #run evaluations for LFRanking
            #run LFRanking algorithm
            LFRanking, pathLFRanking = runLFRanking(originalRanking, protected, nonProtected, 4, dataSetName) 
            #create CSV file with ranking outputs
            createRankingCSV(LFRanking, pathLFRanking,k)
            #Update the currentIndex of a candidate according to LFRanking
            LFRanking = updateCurrentIndex(LFRanking)
            #evaluate LFRanking with all available measures
            evalResults += (runMetrics(evalK, protected, nonProtected, LFRanking, originalRanking, dataSetName, 'LFRanking'))
        except Exception:
            print('Could not create LFRanking for ' + dataSetName)
            pass
    
    return evalResults


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

def updateLearnedIndex(ranking):
    
    """
    Updates the learnedIndex of a ranking according to the current order in the
    list
    @param ranking: list with candidates of a ranking
    
    return list of candidates with updated learnedIndex according to their 
    position in the current ranking
    
    """
    
    index = 0
    
    for i in range(len(ranking)):
        index += 1
        
        ranking[i].learnedIndex = index
        
    return ranking

def getDataSetName(fileName):
    
    """
    Extracts name of file for score based eval
    
    @param fileName: Name of the file with .csv ending
    
    return fileName without .csv
    """
    
    name = fileName.split('.')[0]
    
    return name

    
main()