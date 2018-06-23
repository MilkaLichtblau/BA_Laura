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
            listResults, listFileNames = evaluateLearning(listNetRanking, dataSetName, queryNumbers, True)
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
    
    #results = [['TREC_49.0', 'ListNet', 'AP', 0.0], ['TREC_49.0', 'ListNet', 'NDCG@1', 0.05277684584603507], ['TREC_49.0', 'ListNet', 'NDCG@5', 0.05010285584041623], ['TREC_49.0', 'ListNet', 'NDCG@10', 0.09932490983348058], ['TREC_49.0', 'ListNet', 'rKL', 0.034667738397651075], ['TREC_49.0', 'ListNet', 'DTR', 0.7392000561899302], ['TREC_49.0', 'ListNet', 'DIR', 1.5194543482019525], ['TREC_49.0', 'ListNet', 'FairnessAtK', 0.94], ['TREC_49.0', 'Color-Blind', 'AP', 0.0], ['TREC_49.0', 'Color-Blind', 'NDCG@1', 0.05277684584603507], ['TREC_49.0', 'Color-Blind', 'NDCG@5', 0.05010285584041623], ['TREC_49.0', 'Color-Blind', 'NDCG@10', 0.09932490983348058], ['TREC_49.0', 'Color-Blind', 'rKL', 0.034667738397651075], ['TREC_49.0', 'Color-Blind', 'DTR', 0.7392000561899302], ['TREC_49.0', 'Color-Blind', 'DIR', 1.5194543482019525], ['TREC_49.0', 'Color-Blind', 'FairnessAtK', 0.94], ['TREC_49.0', 'FeldmanEtAl', 'AP', 0.0], ['TREC_49.0', 'FeldmanEtAl', 'NDCG@1', 0.8639689896739321], ['TREC_49.0', 'FeldmanEtAl', 'NDCG@5', 0.32788958326922807], ['TREC_49.0', 'FeldmanEtAl', 'NDCG@10', 0.22189558986861524], ['TREC_49.0', 'FeldmanEtAl', 'rKL', 0.08689199927594282], ['TREC_49.0', 'FeldmanEtAl', 'DTR', 0.7392000635796951], ['TREC_49.0', 'FeldmanEtAl', 'DIR', 1.5194543481775886], ['TREC_49.0', 'FeldmanEtAl', 'FairnessAtK', 0.94], ['TREC_49.0', 'FOEIR-DPC', 'AP', 0.0013157894736842105], ['TREC_49.0', 'FOEIR-DPC', 'NDCG@1', 0.05277684584603507], ['TREC_49.0', 'FOEIR-DPC', 'NDCG@5', 0.22615387832756398], ['TREC_49.0', 'FOEIR-DPC', 'NDCG@10', 0.15571757931886496], ['TREC_49.0', 'FOEIR-DPC', 'rKL', 0.10277914254223731], ['TREC_49.0', 'FOEIR-DPC', 'DTR', 0.35908867914391834], ['TREC_49.0', 'FOEIR-DPC', 'DIR', 1.0431352282580872], ['TREC_49.0', 'FOEIR-DPC', 'FairnessAtK', 0.2], ['TREC_49.0', 'FOEIR-DTC', 'AP', 0.0013157894736842105], ['TREC_49.0', 'FOEIR-DTC', 'NDCG@1', 0.05277684584603507], ['TREC_49.0', 'FOEIR-DTC', 'NDCG@5', 0.22615387832756398], ['TREC_49.0', 'FOEIR-DTC', 'NDCG@10', 0.15571757931886496], ['TREC_49.0', 'FOEIR-DTC', 'rKL', 0.10277914254223731], ['TREC_49.0', 'FOEIR-DTC', 'DTR', 0.33840788644217934], ['TREC_49.0', 'FOEIR-DTC', 'DIR', 1.009921034608134], ['TREC_49.0', 'FOEIR-DTC', 'FairnessAtK', 0.2], ['TREC_49.0', 'FOEIR-DIC', 'AP', 0.0013157894736842105], ['TREC_49.0', 'FOEIR-DIC', 'NDCG@1', 0.05277684584603507], ['TREC_49.0', 'FOEIR-DIC', 'NDCG@5', 0.22615387832756398], ['TREC_49.0', 'FOEIR-DIC', 'NDCG@10', 0.15571757931886496], ['TREC_49.0', 'FOEIR-DIC', 'rKL', 0.10277914254223731], ['TREC_49.0', 'FOEIR-DIC', 'DTR', 0.33840788644217934], ['TREC_49.0', 'FOEIR-DIC', 'DIR', 1.009921034608134], ['TREC_49.0', 'FOEIR-DIC', 'FairnessAtK', 0.2], ['TREC_49.0', 'FAIR', 'AP', 0.0], ['TREC_49.0', 'FAIR', 'NDCG@1', 0.05277684584603507], ['TREC_49.0', 'FAIR', 'NDCG@5', 0.22615387832756398], ['TREC_49.0', 'FAIR', 'NDCG@10', 0.15571757931886496], ['TREC_49.0', 'FAIR', 'rKL', 8.550527334971599e-05], ['TREC_49.0', 'FAIR', 'DTR', 0.7392000510276636], ['TREC_49.0', 'FAIR', 'DIR', 1.5194543479404545], ['TREC_49.0', 'FAIR', 'FairnessAtK', 0.94], ['TREC_49.0', 'LFRanking', 'AP', 0.0], ['TREC_49.0', 'LFRanking', 'NDCG@1', 0.0006351759149535386], ['TREC_49.0', 'LFRanking', 'NDCG@5', 0.008463183143678398], ['TREC_49.0', 'LFRanking', 'NDCG@10', 0.015409740390119296], ['TREC_49.0', 'LFRanking', 'rKL', 0.034667738397651075], ['TREC_49.0', 'LFRanking', 'DTR', 0.13335288673806273], ['TREC_49.0', 'LFRanking', 'DIR', 1.749279140830845], ['TREC_49.0', 'LFRanking', 'FairnessAtK', 0.1], ['TREC_50.0', 'ListNet', 'AP', 0.0], ['TREC_50.0', 'ListNet', 'NDCG@1', 0.025072786351302546], ['TREC_50.0', 'ListNet', 'NDCG@5', 0.03369112517799418], ['TREC_50.0', 'ListNet', 'NDCG@10', 0.14772641184626586], ['TREC_50.0', 'ListNet', 'rKL', 0.03376473807244401], ['TREC_50.0', 'ListNet', 'DTR', 0.7815931795347235], ['TREC_50.0', 'ListNet', 'DIR', 1.7637653137669569], ['TREC_50.0', 'ListNet', 'FairnessAtK', 0.4], ['TREC_50.0', 'Color-Blind', 'AP', 0.0], ['TREC_50.0', 'Color-Blind', 'NDCG@1', 0.025072786351302546], ['TREC_50.0', 'Color-Blind', 'NDCG@5', 0.03369112517799418], ['TREC_50.0', 'Color-Blind', 'NDCG@10', 0.14772641184626586], ['TREC_50.0', 'Color-Blind', 'rKL', 0.03376473807244401], ['TREC_50.0', 'Color-Blind', 'DTR', 0.7815931795347235], ['TREC_50.0', 'Color-Blind', 'DIR', 1.7637653137669569], ['TREC_50.0', 'Color-Blind', 'FairnessAtK', 0.4], ['TREC_50.0', 'FeldmanEtAl', 'AP', 0.0], ['TREC_50.0', 'FeldmanEtAl', 'NDCG@1', 0.8841866893739455], ['TREC_50.0', 'FeldmanEtAl', 'NDCG@5', 0.3237141051417265], ['TREC_50.0', 'FeldmanEtAl', 'NDCG@10', 0.2242403973563218], ['TREC_50.0', 'FeldmanEtAl', 'rKL', 0.08318344212615852], ['TREC_50.0', 'FeldmanEtAl', 'DTR', 0.7815931795875366], ['TREC_50.0', 'FeldmanEtAl', 'DIR', 1.763765313766272], ['TREC_50.0', 'FeldmanEtAl', 'FairnessAtK', 0.4], ['TREC_50.0', 'FOEIR-DPC', 'AP', 0.0], ['TREC_50.0', 'FOEIR-DPC', 'NDCG@1', 0.025072786351302546], ['TREC_50.0', 'FOEIR-DPC', 'NDCG@5', 0.21596829027627035], ['TREC_50.0', 'FOEIR-DPC', 'NDCG@10', 0.15415287324719792], ['TREC_50.0', 'FOEIR-DPC', 'rKL', 0.15908273676767326], ['TREC_50.0', 'FOEIR-DPC', 'DTR', 0.4024478604993275], ['TREC_50.0', 'FOEIR-DPC', 'DIR', 1.2780930355559865], ['TREC_50.0', 'FOEIR-DPC', 'FairnessAtK', 0.2], ['TREC_50.0', 'FOEIR-DTC', 'AP', 0.0], ['TREC_50.0', 'FOEIR-DTC', 'NDCG@1', 0.025072786351302546], ['TREC_50.0', 'FOEIR-DTC', 'NDCG@5', 0.21596829027627035], ['TREC_50.0', 'FOEIR-DTC', 'NDCG@10', 0.15415287324719792], ['TREC_50.0', 'FOEIR-DTC', 'rKL', 0.15908273676767326], ['TREC_50.0', 'FOEIR-DTC', 'DTR', 0.34065794301552466], ['TREC_50.0', 'FOEIR-DTC', 'DIR', 1.1091886299109592], ['TREC_50.0', 'FOEIR-DTC', 'FairnessAtK', 0.2], ['TREC_50.0', 'FOEIR-DIC', 'AP', 0.0], ['TREC_50.0', 'FOEIR-DIC', 'NDCG@1', 0.025072786351302546], ['TREC_50.0', 'FOEIR-DIC', 'NDCG@5', 0.21596829027627035], ['TREC_50.0', 'FOEIR-DIC', 'NDCG@10', 0.15415287324719792], ['TREC_50.0', 'FOEIR-DIC', 'rKL', 0.15908273676767326], ['TREC_50.0', 'FOEIR-DIC', 'DTR', 0.34065794301552466], ['TREC_50.0', 'FOEIR-DIC', 'DIR', 1.1091886299109592], ['TREC_50.0', 'FOEIR-DIC', 'FairnessAtK', 0.2], ['TREC_50.0', 'FAIR', 'AP', 0.0], ['TREC_50.0', 'FAIR', 'NDCG@1', 0.025072786351302546], ['TREC_50.0', 'FAIR', 'NDCG@5', 0.21596829027627035], ['TREC_50.0', 'FAIR', 'NDCG@10', 0.15415287324719792], ['TREC_50.0', 'FAIR', 'rKL', 0.0], ['TREC_50.0', 'FAIR', 'DTR', 0.7815931789803742], ['TREC_50.0', 'FAIR', 'DIR', 1.763765313766428], ['TREC_50.0', 'FAIR', 'FairnessAtK', 0.4], ['TREC_50.0', 'LFRanking', 'AP', 0.0], ['TREC_50.0', 'LFRanking', 'NDCG@1', 0.04213581895557913], ['TREC_50.0', 'LFRanking', 'NDCG@5', 0.16647100566184525], ['TREC_50.0', 'LFRanking', 'NDCG@10', 0.12018037301416433], ['TREC_50.0', 'LFRanking', 'rKL', 0.03376473807244401], ['TREC_50.0', 'LFRanking', 'DTR', 0.04191316226892535], ['TREC_50.0', 'LFRanking', 'DIR', 1.8098111746476933], ['TREC_50.0', 'LFRanking', 'FairnessAtK', 0.16], ['TREC_51.0', 'ListNet', 'AP', 0.0], ['TREC_51.0', 'ListNet', 'NDCG@1', 0.19001379410428645], ['TREC_51.0', 'ListNet', 'NDCG@5', 0.7185533117801264], ['TREC_51.0', 'ListNet', 'NDCG@10', 0.8090106961840702], ['TREC_51.0', 'ListNet', 'rKL', 0.03147780059730986], ['TREC_51.0', 'ListNet', 'DTR', 1.1094658065849698], ['TREC_51.0', 'ListNet', 'DIR', 1.5916063947389536], ['TREC_51.0', 'ListNet', 'FairnessAtK', 0.9], ['TREC_51.0', 'Color-Blind', 'AP', 0.0], ['TREC_51.0', 'Color-Blind', 'NDCG@1', 0.19001379410428645], ['TREC_51.0', 'Color-Blind', 'NDCG@5', 0.7185533117801264], ['TREC_51.0', 'Color-Blind', 'NDCG@10', 0.8090106961840702], ['TREC_51.0', 'Color-Blind', 'rKL', 0.03147780059730986], ['TREC_51.0', 'Color-Blind', 'DTR', 1.1094658065849698], ['TREC_51.0', 'Color-Blind', 'DIR', 1.5916063947389536], ['TREC_51.0', 'Color-Blind', 'FairnessAtK', 0.9], ['TREC_51.0', 'FeldmanEtAl', 'AP', 0.0], ['TREC_51.0', 'FeldmanEtAl', 'NDCG@1', 0.19001379410428645], ['TREC_51.0', 'FeldmanEtAl', 'NDCG@5', 0.7185533117801264], ['TREC_51.0', 'FeldmanEtAl', 'NDCG@10', 0.8090106961840702], ['TREC_51.0', 'FeldmanEtAl', 'rKL', 0.08289145059562968], ['TREC_51.0', 'FeldmanEtAl', 'DTR', 1.1094658065849698], ['TREC_51.0', 'FeldmanEtAl', 'DIR', 1.5916063947389536], ['TREC_51.0', 'FeldmanEtAl', 'FairnessAtK', 0.9], ['TREC_51.0', 'FOEIR-DPC', 'AP', 0.0], ['TREC_51.0', 'FOEIR-DPC', 'NDCG@1', 0.19001379410428645], ['TREC_51.0', 'FOEIR-DPC', 'NDCG@5', 0.7185533117801264], ['TREC_51.0', 'FOEIR-DPC', 'NDCG@10', 0.8090106961840702], ['TREC_51.0', 'FOEIR-DPC', 'rKL', 0.1476133180651315], ['TREC_51.0', 'FOEIR-DPC', 'DTR', 0.6332054634254005], ['TREC_51.0', 'FOEIR-DPC', 'DIR', 0.9945497417809384], ['TREC_51.0', 'FOEIR-DPC', 'FairnessAtK', 1.0], ['TREC_51.0', 'FOEIR-DTC', 'AP', 0.0], ['TREC_51.0', 'FOEIR-DTC', 'NDCG@1', 0.19001379410428645], ['TREC_51.0', 'FOEIR-DTC', 'NDCG@5', 0.7185533117801264], ['TREC_51.0', 'FOEIR-DTC', 'NDCG@10', 0.8090106961840702], ['TREC_51.0', 'FOEIR-DTC', 'rKL', 0.1476133180651315], ['TREC_51.0', 'FOEIR-DTC', 'DTR', 0.6332054634254005], ['TREC_51.0', 'FOEIR-DTC', 'DIR', 0.9945497417809384], ['TREC_51.0', 'FOEIR-DTC', 'FairnessAtK', 1.0], ['TREC_51.0', 'FOEIR-DIC', 'AP', 0.0], ['TREC_51.0', 'FOEIR-DIC', 'NDCG@1', 0.19001379410428645], ['TREC_51.0', 'FOEIR-DIC', 'NDCG@5', 0.7185533117801264], ['TREC_51.0', 'FOEIR-DIC', 'NDCG@10', 0.8090106961840702], ['TREC_51.0', 'FOEIR-DIC', 'rKL', 0.1476133180651315], ['TREC_51.0', 'FOEIR-DIC', 'DTR', 0.6332054634254005], ['TREC_51.0', 'FOEIR-DIC', 'DIR', 0.9945497417809384], ['TREC_51.0', 'FOEIR-DIC', 'FairnessAtK', 1.0], ['TREC_51.0', 'FAIR', 'AP', 0.0], ['TREC_51.0', 'FAIR', 'NDCG@1', 0.19001379410428645], ['TREC_51.0', 'FAIR', 'NDCG@5', 0.7185533117801264], ['TREC_51.0', 'FAIR', 'NDCG@10', 0.8090106961840702], ['TREC_51.0', 'FAIR', 'rKL', 0.0], ['TREC_51.0', 'FAIR', 'DTR', 1.1094658065849698], ['TREC_51.0', 'FAIR', 'DIR', 1.5916063947389536], ['TREC_51.0', 'FAIR', 'FairnessAtK', 0.9], ['TREC_51.0', 'LFRanking', 'AP', 0.0], ['TREC_51.0', 'LFRanking', 'NDCG@1', 0.04213581895557913], ['TREC_51.0', 'LFRanking', 'NDCG@5', 0.022728844009491867], ['TREC_51.0', 'LFRanking', 'NDCG@10', 0.08222192174921601], ['TREC_51.0', 'LFRanking', 'rKL', 0.03147780059730986], ['TREC_51.0', 'LFRanking', 'DTR', 0.10301381498550863], ['TREC_51.0', 'LFRanking', 'DIR', 1.6616416221953552], ['TREC_51.0', 'LFRanking', 'FairnessAtK', 0.73], ['TREC_52.0', 'ListNet', 'AP', 0.0], ['TREC_52.0', 'ListNet', 'NDCG@1', 0.8568852116770033], ['TREC_52.0', 'ListNet', 'NDCG@5', 0.7457911089165385], ['TREC_52.0', 'ListNet', 'NDCG@10', 0.830295325350454], ['TREC_52.0', 'ListNet', 'rKL', 0.03605304040080836], ['TREC_52.0', 'ListNet', 'DTR', 1.3179894544698068], ['TREC_52.0', 'ListNet', 'DIR', 1.5782775657102512], ['TREC_52.0', 'ListNet', 'FairnessAtK', 0.94], ['TREC_52.0', 'Color-Blind', 'AP', 0.0], ['TREC_52.0', 'Color-Blind', 'NDCG@1', 0.8568852116770033], ['TREC_52.0', 'Color-Blind', 'NDCG@5', 0.7457911089165385], ['TREC_52.0', 'Color-Blind', 'NDCG@10', 0.830295325350454], ['TREC_52.0', 'Color-Blind', 'rKL', 0.03605304040080836], ['TREC_52.0', 'Color-Blind', 'DTR', 1.3179894544698068], ['TREC_52.0', 'Color-Blind', 'DIR', 1.5782775657102512], ['TREC_52.0', 'Color-Blind', 'FairnessAtK', 0.94], ['TREC_52.0', 'FeldmanEtAl', 'AP', 0.0], ['TREC_52.0', 'FeldmanEtAl', 'NDCG@1', 0.8568852116770033], ['TREC_52.0', 'FeldmanEtAl', 'NDCG@5', 0.7457911089165385], ['TREC_52.0', 'FeldmanEtAl', 'NDCG@10', 0.830295325350454], ['TREC_52.0', 'FeldmanEtAl', 'rKL', 0.08728060955662105], ['TREC_52.0', 'FeldmanEtAl', 'DTR', 1.3179894544698068], ['TREC_52.0', 'FeldmanEtAl', 'DIR', 1.5782775657102512], ['TREC_52.0', 'FeldmanEtAl', 'FairnessAtK', 0.94], ['TREC_52.0', 'FOEIR-DPC', 'AP', 0.0], ['TREC_52.0', 'FOEIR-DPC', 'NDCG@1', 0.8568852116770033], ['TREC_52.0', 'FOEIR-DPC', 'NDCG@5', 0.7457911089165385], ['TREC_52.0', 'FOEIR-DPC', 'NDCG@10', 0.830295325350454], ['TREC_52.0', 'FOEIR-DPC', 'rKL', 0.16986408493530886], ['TREC_52.0', 'FOEIR-DPC', 'DTR', 0.8161828826270046], ['TREC_52.0', 'FOEIR-DPC', 'DIR', 1.0416991550888237], ['TREC_52.0', 'FOEIR-DPC', 'FairnessAtK', 1.0], ['TREC_52.0', 'FOEIR-DTC', 'AP', 0.0], ['TREC_52.0', 'FOEIR-DTC', 'NDCG@1', 0.8568852116770033], ['TREC_52.0', 'FOEIR-DTC', 'NDCG@5', 0.7457911089165385], ['TREC_52.0', 'FOEIR-DTC', 'NDCG@10', 0.830295325350454], ['TREC_52.0', 'FOEIR-DTC', 'rKL', 0.16986408493530886], ['TREC_52.0', 'FOEIR-DTC', 'DTR', 0.8161828826270046], ['TREC_52.0', 'FOEIR-DTC', 'DIR', 1.0416991550888237], ['TREC_52.0', 'FOEIR-DTC', 'FairnessAtK', 1.0], ['TREC_52.0', 'FOEIR-DIC', 'AP', 0.0], ['TREC_52.0', 'FOEIR-DIC', 'NDCG@1', 0.8568852116770033], ['TREC_52.0', 'FOEIR-DIC', 'NDCG@5', 0.7457911089165385], ['TREC_52.0', 'FOEIR-DIC', 'NDCG@10', 0.830295325350454], ['TREC_52.0', 'FOEIR-DIC', 'rKL', 0.16986408493530886], ['TREC_52.0', 'FOEIR-DIC', 'DTR', 0.8161828826270046], ['TREC_52.0', 'FOEIR-DIC', 'DIR', 1.0416991550888237], ['TREC_52.0', 'FOEIR-DIC', 'FairnessAtK', 1.0], ['TREC_52.0', 'FAIR', 'AP', 0.0], ['TREC_52.0', 'FAIR', 'NDCG@1', 0.8568852116770033], ['TREC_52.0', 'FAIR', 'NDCG@5', 0.7457911089165385], ['TREC_52.0', 'FAIR', 'NDCG@10', 0.830295325350454], ['TREC_52.0', 'FAIR', 'rKL', 9.8134436396836e-06], ['TREC_52.0', 'FAIR', 'DTR', 1.3179894544698068], ['TREC_52.0', 'FAIR', 'DIR', 1.5782775657102512], ['TREC_52.0', 'FAIR', 'FairnessAtK', 0.94], ['TREC_52.0', 'LFRanking', 'AP', 0.0], ['TREC_52.0', 'LFRanking', 'NDCG@1', 0.06082877262494944], ['TREC_52.0', 'LFRanking', 'NDCG@5', 0.07119598782843108], ['TREC_52.0', 'LFRanking', 'NDCG@10', 0.07343733923333728], ['TREC_52.0', 'LFRanking', 'rKL', 0.03605304040080836], ['TREC_52.0', 'LFRanking', 'DTR', 0.17899528591984948], ['TREC_52.0', 'LFRanking', 'DIR', 1.7754590122084273], ['TREC_52.0', 'LFRanking', 'FairnessAtK', 0.1], ['TREC_53.0', 'ListNet', 'AP', 0.0], ['TREC_53.0', 'ListNet', 'NDCG@1', 0.055454015943723865], ['TREC_53.0', 'ListNet', 'NDCG@5', 0.4852848127850224], ['TREC_53.0', 'ListNet', 'NDCG@10', 0.6538287621379423], ['TREC_53.0', 'ListNet', 'rKL', 0.037062329619373495], ['TREC_53.0', 'ListNet', 'DTR', 1.2956556442684866], ['TREC_53.0', 'ListNet', 'DIR', 1.6783445594291382], ['TREC_53.0', 'ListNet', 'FairnessAtK', 0.69], ['TREC_53.0', 'Color-Blind', 'AP', 0.0], ['TREC_53.0', 'Color-Blind', 'NDCG@1', 0.055454015943723865], ['TREC_53.0', 'Color-Blind', 'NDCG@5', 0.4852848127850224], ['TREC_53.0', 'Color-Blind', 'NDCG@10', 0.6538287621379423], ['TREC_53.0', 'Color-Blind', 'rKL', 0.037062329619373495], ['TREC_53.0', 'Color-Blind', 'DTR', 1.2956556442684866], ['TREC_53.0', 'Color-Blind', 'DIR', 1.6783445594291382], ['TREC_53.0', 'Color-Blind', 'FairnessAtK', 0.69], ['TREC_53.0', 'FeldmanEtAl', 'AP', 0.0], ['TREC_53.0', 'FeldmanEtAl', 'NDCG@1', 0.055454015943723865], ['TREC_53.0', 'FeldmanEtAl', 'NDCG@5', 0.4852848127850224], ['TREC_53.0', 'FeldmanEtAl', 'NDCG@10', 0.6538287621379423], ['TREC_53.0', 'FeldmanEtAl', 'rKL', 0.09230044602215362], ['TREC_53.0', 'FeldmanEtAl', 'DTR', 1.2956556442684866], ['TREC_53.0', 'FeldmanEtAl', 'DIR', 1.6783445594291382], ['TREC_53.0', 'FeldmanEtAl', 'FairnessAtK', 0.69], ['TREC_53.0', 'FOEIR-DPC', 'AP', 0.0], ['TREC_53.0', 'FOEIR-DPC', 'NDCG@1', 0.055454015943723865], ['TREC_53.0', 'FOEIR-DPC', 'NDCG@5', 0.4852848127850224], ['TREC_53.0', 'FOEIR-DPC', 'NDCG@10', 0.6538287621379423], ['TREC_53.0', 'FOEIR-DPC', 'rKL', 0.17461919999338835], ['TREC_53.0', 'FOEIR-DPC', 'DTR', 0.6777200522078111], ['TREC_53.0', 'FOEIR-DPC', 'DIR', 0.991627162306643], ['TREC_53.0', 'FOEIR-DPC', 'FairnessAtK', 0.95], ['TREC_53.0', 'FOEIR-DTC', 'AP', 0.0], ['TREC_53.0', 'FOEIR-DTC', 'NDCG@1', 0.055454015943723865], ['TREC_53.0', 'FOEIR-DTC', 'NDCG@5', 0.4852848127850224], ['TREC_53.0', 'FOEIR-DTC', 'NDCG@10', 0.6538287621379423], ['TREC_53.0', 'FOEIR-DTC', 'rKL', 0.17461919999338835], ['TREC_53.0', 'FOEIR-DTC', 'DTR', 0.6777200522078111], ['TREC_53.0', 'FOEIR-DTC', 'DIR', 0.991627162306643], ['TREC_53.0', 'FOEIR-DTC', 'FairnessAtK', 0.95], ['TREC_53.0', 'FOEIR-DIC', 'AP', 0.0], ['TREC_53.0', 'FOEIR-DIC', 'NDCG@1', 0.055454015943723865], ['TREC_53.0', 'FOEIR-DIC', 'NDCG@5', 0.4852848127850224], ['TREC_53.0', 'FOEIR-DIC', 'NDCG@10', 0.6538287621379423], ['TREC_53.0', 'FOEIR-DIC', 'rKL', 0.17461919999338835], ['TREC_53.0', 'FOEIR-DIC', 'DTR', 0.6777200522078111], ['TREC_53.0', 'FOEIR-DIC', 'DIR', 0.991627162306643], ['TREC_53.0', 'FOEIR-DIC', 'FairnessAtK', 0.95], ['TREC_53.0', 'FAIR', 'AP', 0.0], ['TREC_53.0', 'FAIR', 'NDCG@1', 0.055454015943723865], ['TREC_53.0', 'FAIR', 'NDCG@5', 0.4852848127850224], ['TREC_53.0', 'FAIR', 'NDCG@10', 0.6538287621379423], ['TREC_53.0', 'FAIR', 'rKL', 0.0], ['TREC_53.0', 'FAIR', 'DTR', 1.2956556442684866], ['TREC_53.0', 'FAIR', 'DIR', 1.6783445594291382], ['TREC_53.0', 'FAIR', 'FairnessAtK', 0.69], ['TREC_53.0', 'LFRanking', 'AP', 0.0], ['TREC_53.0', 'LFRanking', 'NDCG@1', 0.21135603081338444], ['TREC_53.0', 'LFRanking', 'NDCG@5', 0.276836229665186], ['TREC_53.0', 'LFRanking', 'NDCG@10', 0.2060704442517468], ['TREC_53.0', 'LFRanking', 'rKL', 0.037062329619373495], ['TREC_53.0', 'LFRanking', 'DTR', 0.21223437522706293], ['TREC_53.0', 'LFRanking', 'DIR', 1.7560588493559468], ['TREC_53.0', 'LFRanking', 'FairnessAtK', 0.35], ['TREC_54.0', 'ListNet', 'AP', 0.0], ['TREC_54.0', 'ListNet', 'NDCG@1', 0.0019067368111595926], ['TREC_54.0', 'ListNet', 'NDCG@5', 0.00837521254198428], ['TREC_54.0', 'ListNet', 'NDCG@10', 0.2096582865955668], ['TREC_54.0', 'ListNet', 'rKL', 0.034988128506753494], ['TREC_54.0', 'ListNet', 'DTR', 1.0699883131441081], ['TREC_54.0', 'ListNet', 'DIR', 1.890308349849457], ['TREC_54.0', 'ListNet', 'FairnessAtK', 0.35], ['TREC_54.0', 'Color-Blind', 'AP', 0.0], ['TREC_54.0', 'Color-Blind', 'NDCG@1', 0.0019067368111595926], ['TREC_54.0', 'Color-Blind', 'NDCG@5', 0.00837521254198428], ['TREC_54.0', 'Color-Blind', 'NDCG@10', 0.2096582865955668], ['TREC_54.0', 'Color-Blind', 'rKL', 0.034988128506753494], ['TREC_54.0', 'Color-Blind', 'DTR', 1.0699883131441081], ['TREC_54.0', 'Color-Blind', 'DIR', 1.890308349849457], ['TREC_54.0', 'Color-Blind', 'FairnessAtK', 0.35], ['TREC_54.0', 'FeldmanEtAl', 'AP', 0.0], ['TREC_54.0', 'FeldmanEtAl', 'NDCG@1', 0.0019067368111595926], ['TREC_54.0', 'FeldmanEtAl', 'NDCG@5', 0.00837521254198428], ['TREC_54.0', 'FeldmanEtAl', 'NDCG@10', 0.2096582865955668], ['TREC_54.0', 'FeldmanEtAl', 'rKL', 0.08264472302436278], ['TREC_54.0', 'FeldmanEtAl', 'DTR', 1.0699883131441081], ['TREC_54.0', 'FeldmanEtAl', 'DIR', 1.890308349849457], ['TREC_54.0', 'FeldmanEtAl', 'FairnessAtK', 0.35], ['TREC_54.0', 'FOEIR-DPC', 'AP', 0.0], ['TREC_54.0', 'FOEIR-DPC', 'NDCG@1', 0.0019067368111595926], ['TREC_54.0', 'FOEIR-DPC', 'NDCG@5', 0.00837521254198428], ['TREC_54.0', 'FOEIR-DPC', 'NDCG@10', 0.2096582865955668], ['TREC_54.0', 'FOEIR-DPC', 'rKL', 0.164846599548926], ['TREC_54.0', 'FOEIR-DPC', 'DTR', 0.4332310938369232], ['TREC_54.0', 'FOEIR-DPC', 'DIR', 0.9892315678423922], ['TREC_54.0', 'FOEIR-DPC', 'FairnessAtK', 0.1], ['TREC_54.0', 'FOEIR-DTC', 'AP', 0.0], ['TREC_54.0', 'FOEIR-DTC', 'NDCG@1', 0.0019067368111595926], ['TREC_54.0', 'FOEIR-DTC', 'NDCG@5', 0.00837521254198428], ['TREC_54.0', 'FOEIR-DTC', 'NDCG@10', 0.2096582865955668], ['TREC_54.0', 'FOEIR-DTC', 'rKL', 0.164846599548926], ['TREC_54.0', 'FOEIR-DTC', 'DTR', 0.4332310938369232], ['TREC_54.0', 'FOEIR-DTC', 'DIR', 0.9892315678423922], ['TREC_54.0', 'FOEIR-DTC', 'FairnessAtK', 0.1], ['TREC_54.0', 'FOEIR-DIC', 'AP', 0.0], ['TREC_54.0', 'FOEIR-DIC', 'NDCG@1', 0.0019067368111595926], ['TREC_54.0', 'FOEIR-DIC', 'NDCG@5', 0.00837521254198428], ['TREC_54.0', 'FOEIR-DIC', 'NDCG@10', 0.2096582865955668], ['TREC_54.0', 'FOEIR-DIC', 'rKL', 0.164846599548926], ['TREC_54.0', 'FOEIR-DIC', 'DTR', 0.4332310938369232], ['TREC_54.0', 'FOEIR-DIC', 'DIR', 0.9892315678423922], ['TREC_54.0', 'FOEIR-DIC', 'FairnessAtK', 0.1], ['TREC_54.0', 'FAIR', 'AP', 0.0], ['TREC_54.0', 'FAIR', 'NDCG@1', 0.0019067368111595926], ['TREC_54.0', 'FAIR', 'NDCG@5', 0.00837521254198428], ['TREC_54.0', 'FAIR', 'NDCG@10', 0.2096582865955668], ['TREC_54.0', 'FAIR', 'rKL', 0.0], ['TREC_54.0', 'FAIR', 'DTR', 1.0699883131441081], ['TREC_54.0', 'FAIR', 'DIR', 1.890308349849457], ['TREC_54.0', 'FAIR', 'FairnessAtK', 0.35], ['TREC_54.0', 'LFRanking', 'AP', 0.0], ['TREC_54.0', 'LFRanking', 'NDCG@1', 0.028985475110228705], ['TREC_54.0', 'LFRanking', 'NDCG@5', 0.034522038572703], ['TREC_54.0', 'LFRanking', 'NDCG@10', 0.056401393191249614], ['TREC_54.0', 'LFRanking', 'rKL', 0.034988128506753494], ['TREC_54.0', 'LFRanking', 'DTR', 0.1874265935610421], ['TREC_54.0', 'LFRanking', 'DIR', 1.7247195444935468], ['TREC_54.0', 'LFRanking', 'FairnessAtK', 0.61], ['TREC_55.0', 'ListNet', 'AP', 0.0], ['TREC_55.0', 'ListNet', 'NDCG@1', 0.0006351759149535386], ['TREC_55.0', 'ListNet', 'NDCG@5', 0.44630481416164414], ['TREC_55.0', 'ListNet', 'NDCG@10', 0.6354826272405858], ['TREC_55.0', 'ListNet', 'rKL', 0.03259418400472522], ['TREC_55.0', 'ListNet', 'DTR', 1.2982198592219327], ['TREC_55.0', 'ListNet', 'DIR', 1.5848062078546274], ['TREC_55.0', 'ListNet', 'FairnessAtK', 0.9], ['TREC_55.0', 'Color-Blind', 'AP', 0.0], ['TREC_55.0', 'Color-Blind', 'NDCG@1', 0.0006351759149535386], ['TREC_55.0', 'Color-Blind', 'NDCG@5', 0.44630481416164414], ['TREC_55.0', 'Color-Blind', 'NDCG@10', 0.6354826272405858], ['TREC_55.0', 'Color-Blind', 'rKL', 0.03259418400472522], ['TREC_55.0', 'Color-Blind', 'DTR', 1.2982198592219327], ['TREC_55.0', 'Color-Blind', 'DIR', 1.5848062078546274], ['TREC_55.0', 'Color-Blind', 'FairnessAtK', 0.9], ['TREC_55.0', 'FeldmanEtAl', 'AP', 0.0], ['TREC_55.0', 'FeldmanEtAl', 'NDCG@1', 0.0006351759149535386], ['TREC_55.0', 'FeldmanEtAl', 'NDCG@5', 0.44630481416164414], ['TREC_55.0', 'FeldmanEtAl', 'NDCG@10', 0.6354826272405858], ['TREC_55.0', 'FeldmanEtAl', 'rKL', 0.08173814222305387], ['TREC_55.0', 'FeldmanEtAl', 'DTR', 1.2982198592219327], ['TREC_55.0', 'FeldmanEtAl', 'DIR', 1.5848062078546274], ['TREC_55.0', 'FeldmanEtAl', 'FairnessAtK', 0.9], ['TREC_55.0', 'FOEIR-DPC', 'AP', 0.0], ['TREC_55.0', 'FOEIR-DPC', 'NDCG@1', 0.0006351759149535386], ['TREC_55.0', 'FOEIR-DPC', 'NDCG@5', 0.44630481416164414], ['TREC_55.0', 'FOEIR-DPC', 'NDCG@10', 0.6354826272405858], ['TREC_55.0', 'FOEIR-DPC', 'rKL', 0.1526028020037078], ['TREC_55.0', 'FOEIR-DPC', 'DTR', 0.703236883431392], ['TREC_55.0', 'FOEIR-DPC', 'DIR', 0.9965856757166465], ['TREC_55.0', 'FOEIR-DPC', 'FairnessAtK', 1.0], ['TREC_55.0', 'FOEIR-DTC', 'AP', 0.0], ['TREC_55.0', 'FOEIR-DTC', 'NDCG@1', 0.0006351759149535386], ['TREC_55.0', 'FOEIR-DTC', 'NDCG@5', 0.44630481416164414], ['TREC_55.0', 'FOEIR-DTC', 'NDCG@10', 0.6354826272405858], ['TREC_55.0', 'FOEIR-DTC', 'rKL', 0.1526028020037078], ['TREC_55.0', 'FOEIR-DTC', 'DTR', 0.703236883431392], ['TREC_55.0', 'FOEIR-DTC', 'DIR', 0.9965856757166465], ['TREC_55.0', 'FOEIR-DTC', 'FairnessAtK', 1.0], ['TREC_55.0', 'FOEIR-DIC', 'AP', 0.0], ['TREC_55.0', 'FOEIR-DIC', 'NDCG@1', 0.0006351759149535386], ['TREC_55.0', 'FOEIR-DIC', 'NDCG@5', 0.44630481416164414], ['TREC_55.0', 'FOEIR-DIC', 'NDCG@10', 0.6354826272405858], ['TREC_55.0', 'FOEIR-DIC', 'rKL', 0.1526028020037078], ['TREC_55.0', 'FOEIR-DIC', 'DTR', 0.703236883431392], ['TREC_55.0', 'FOEIR-DIC', 'DIR', 0.9965856757166465], ['TREC_55.0', 'FOEIR-DIC', 'FairnessAtK', 1.0], ['TREC_55.0', 'FAIR', 'AP', 0.0], ['TREC_55.0', 'FAIR', 'NDCG@1', 0.0006351759149535386], ['TREC_55.0', 'FAIR', 'NDCG@5', 0.44630481416164414], ['TREC_55.0', 'FAIR', 'NDCG@10', 0.6354826272405858], ['TREC_55.0', 'FAIR', 'rKL', 0.0], ['TREC_55.0', 'FAIR', 'DTR', 1.2982198592219327], ['TREC_55.0', 'FAIR', 'DIR', 1.5848062078546274], ['TREC_55.0', 'FAIR', 'FairnessAtK', 0.9], ['TREC_55.0', 'LFRanking', 'AP', 0.0], ['TREC_55.0', 'LFRanking', 'NDCG@1', 0.04877380253713089], ['TREC_55.0', 'LFRanking', 'NDCG@5', 0.2020215664321463], ['TREC_55.0', 'LFRanking', 'NDCG@10', 0.21426969297869083], ['TREC_55.0', 'LFRanking', 'rKL', 0.03259418400472522], ['TREC_55.0', 'LFRanking', 'DTR', 0.7181965777630588], ['TREC_55.0', 'LFRanking', 'DIR', 1.7712519684771044], ['TREC_55.0', 'LFRanking', 'FairnessAtK', 0.35], ['TREC_56.0', 'ListNet', 'AP', 0.00125], ['TREC_56.0', 'ListNet', 'NDCG@1', 0.0006351759149535386], ['TREC_56.0', 'ListNet', 'NDCG@5', 0.05505755753370157], ['TREC_56.0', 'ListNet', 'NDCG@10', 0.30636679960740937], ['TREC_56.0', 'ListNet', 'rKL', 0.03837635328901292], ['TREC_56.0', 'ListNet', 'DTR', 1.0397669100929738], ['TREC_56.0', 'ListNet', 'DIR', 1.5350425979782893], ['TREC_56.0', 'ListNet', 'FairnessAtK', 0.98], ['TREC_56.0', 'Color-Blind', 'AP', 0.00125], ['TREC_56.0', 'Color-Blind', 'NDCG@1', 0.0006351759149535386], ['TREC_56.0', 'Color-Blind', 'NDCG@5', 0.05505755753370157], ['TREC_56.0', 'Color-Blind', 'NDCG@10', 0.30636679960740937], ['TREC_56.0', 'Color-Blind', 'rKL', 0.03837635328901292], ['TREC_56.0', 'Color-Blind', 'DTR', 1.0397669100929738], ['TREC_56.0', 'Color-Blind', 'DIR', 1.5350425979782893], ['TREC_56.0', 'Color-Blind', 'FairnessAtK', 0.98], ['TREC_56.0', 'FeldmanEtAl', 'AP', 0.00125], ['TREC_56.0', 'FeldmanEtAl', 'NDCG@1', 0.0006351759149535386], ['TREC_56.0', 'FeldmanEtAl', 'NDCG@5', 0.05505755753370157], ['TREC_56.0', 'FeldmanEtAl', 'NDCG@10', 0.30636679960740937], ['TREC_56.0', 'FeldmanEtAl', 'rKL', 0.10102689842749057], ['TREC_56.0', 'FeldmanEtAl', 'DTR', 1.0397669100929738], ['TREC_56.0', 'FeldmanEtAl', 'DIR', 1.5350425979782893], ['TREC_56.0', 'FeldmanEtAl', 'FairnessAtK', 0.98], ['TREC_56.0', 'FOEIR-DPC', 'AP', 0.003125], ['TREC_56.0', 'FOEIR-DPC', 'NDCG@1', 0.0006351759149535386], ['TREC_56.0', 'FOEIR-DPC', 'NDCG@5', 0.05505755753370157], ['TREC_56.0', 'FOEIR-DPC', 'NDCG@10', 0.30636679960740937], ['TREC_56.0', 'FOEIR-DPC', 'rKL', 0.1803270896369602], ['TREC_56.0', 'FOEIR-DPC', 'DTR', 0.49313809460824953], ['TREC_56.0', 'FOEIR-DPC', 'DIR', 0.9654074476322888], ['TREC_56.0', 'FOEIR-DPC', 'FairnessAtK', 0.1], ['TREC_56.0', 'FOEIR-DTC', 'AP', 0.003125], ['TREC_56.0', 'FOEIR-DTC', 'NDCG@1', 0.0006351759149535386], ['TREC_56.0', 'FOEIR-DTC', 'NDCG@5', 0.05505755753370157], ['TREC_56.0', 'FOEIR-DTC', 'NDCG@10', 0.30636679960740937], ['TREC_56.0', 'FOEIR-DTC', 'rKL', 0.1803270896369602], ['TREC_56.0', 'FOEIR-DTC', 'DTR', 0.49313809460824953], ['TREC_56.0', 'FOEIR-DTC', 'DIR', 0.9654074476322888], ['TREC_56.0', 'FOEIR-DTC', 'FairnessAtK', 0.1], ['TREC_56.0', 'FOEIR-DIC', 'AP', 0.003125], ['TREC_56.0', 'FOEIR-DIC', 'NDCG@1', 0.0006351759149535386], ['TREC_56.0', 'FOEIR-DIC', 'NDCG@5', 0.05505755753370157], ['TREC_56.0', 'FOEIR-DIC', 'NDCG@10', 0.30636679960740937], ['TREC_56.0', 'FOEIR-DIC', 'rKL', 0.1803270896369602], ['TREC_56.0', 'FOEIR-DIC', 'DTR', 0.49313809460824953], ['TREC_56.0', 'FOEIR-DIC', 'DIR', 0.9654074476322888], ['TREC_56.0', 'FOEIR-DIC', 'FairnessAtK', 0.1], ['TREC_56.0', 'FAIR', 'AP', 0.00125], ['TREC_56.0', 'FAIR', 'NDCG@1', 0.0006351759149535386], ['TREC_56.0', 'FAIR', 'NDCG@5', 0.05505755753370157], ['TREC_56.0', 'FAIR', 'NDCG@10', 0.30636679960740937], ['TREC_56.0', 'FAIR', 'rKL', 0.0], ['TREC_56.0', 'FAIR', 'DTR', 1.0397669100929738], ['TREC_56.0', 'FAIR', 'DIR', 1.5350425979782893], ['TREC_56.0', 'FAIR', 'FairnessAtK', 0.98], ['TREC_56.0', 'LFRanking', 'AP', 0.0], ['TREC_56.0', 'LFRanking', 'NDCG@1', 0.05344550183782405], ['TREC_56.0', 'LFRanking', 'NDCG@5', 0.22837543901799662], ['TREC_56.0', 'LFRanking', 'NDCG@10', 0.2799160994622843], ['TREC_56.0', 'LFRanking', 'rKL', 0.03837635328901292], ['TREC_56.0', 'LFRanking', 'DTR', 0.03433435450405649], ['TREC_56.0', 'LFRanking', 'DIR', 1.7139619594702324], ['TREC_56.0', 'LFRanking', 'FairnessAtK', 0.31], ['TREC_57.0', 'ListNet', 'AP', 0.0], ['TREC_57.0', 'ListNet', 'NDCG@1', 0.9987312583853045], ['TREC_57.0', 'ListNet', 'NDCG@5', 0.9747979095535096], ['TREC_57.0', 'ListNet', 'NDCG@10', 0.9124817537322476], ['TREC_57.0', 'ListNet', 'rKL', 0.03168266791057939], ['TREC_57.0', 'ListNet', 'DTR', 0.23411726727042298], ['TREC_57.0', 'ListNet', 'DIR', 1.8308180679334185], ['TREC_57.0', 'ListNet', 'FairnessAtK', 0.4], ['TREC_57.0', 'Color-Blind', 'AP', 0.0], ['TREC_57.0', 'Color-Blind', 'NDCG@1', 0.9987312583853045], ['TREC_57.0', 'Color-Blind', 'NDCG@5', 0.9747979095535096], ['TREC_57.0', 'Color-Blind', 'NDCG@10', 0.9124817537322476], ['TREC_57.0', 'Color-Blind', 'rKL', 0.03168266791057939], ['TREC_57.0', 'Color-Blind', 'DTR', 0.23411726727042298], ['TREC_57.0', 'Color-Blind', 'DIR', 1.8308180679334185], ['TREC_57.0', 'Color-Blind', 'FairnessAtK', 0.4], ['TREC_57.0', 'FeldmanEtAl', 'AP', 0.0], ['TREC_57.0', 'FeldmanEtAl', 'NDCG@1', 0.9987312583853045], ['TREC_57.0', 'FeldmanEtAl', 'NDCG@5', 0.9747979095535096], ['TREC_57.0', 'FeldmanEtAl', 'NDCG@10', 0.9124817537322476], ['TREC_57.0', 'FeldmanEtAl', 'rKL', 0.07897109645613859], ['TREC_57.0', 'FeldmanEtAl', 'DTR', 0.23411726727042298], ['TREC_57.0', 'FeldmanEtAl', 'DIR', 1.8308180679334185], ['TREC_57.0', 'FeldmanEtAl', 'FairnessAtK', 0.4], ['TREC_57.0', 'FAIR', 'AP', 0.0], ['TREC_57.0', 'FAIR', 'NDCG@1', 0.9987312583853045], ['TREC_57.0', 'FAIR', 'NDCG@5', 0.9747979095535096], ['TREC_57.0', 'FAIR', 'NDCG@10', 0.9124817537322476], ['TREC_57.0', 'FAIR', 'rKL', 0.0], ['TREC_57.0', 'FAIR', 'DTR', 0.23411726727042298], ['TREC_57.0', 'FAIR', 'DIR', 1.8308180679334185], ['TREC_57.0', 'FAIR', 'FairnessAtK', 0.4], ['TREC_57.0', 'LFRanking', 'AP', 0.0], ['TREC_57.0', 'LFRanking', 'NDCG@1', 0.03488249389595517], ['TREC_57.0', 'LFRanking', 'NDCG@5', 0.03029316154428355], ['TREC_57.0', 'LFRanking', 'NDCG@10', 0.03793308252882508], ['TREC_57.0', 'LFRanking', 'rKL', 0.03168266791057939], ['TREC_57.0', 'LFRanking', 'DTR', 0.13111221766579664], ['TREC_57.0', 'LFRanking', 'DIR', 1.7448236635286152], ['TREC_57.0', 'LFRanking', 'FairnessAtK', 0.1], ['TREC_58.0', 'ListNet', 'AP', 0.0], ['TREC_58.0', 'ListNet', 'NDCG@1', 0.07711762358266568], ['TREC_58.0', 'ListNet', 'NDCG@5', 0.16268736658638386], ['TREC_58.0', 'ListNet', 'NDCG@10', 0.18169393880276363], ['TREC_58.0', 'ListNet', 'rKL', 0.03161110549613829], ['TREC_58.0', 'ListNet', 'DTR', 0.682829941302739], ['TREC_58.0', 'ListNet', 'DIR', 1.567228463800318], ['TREC_58.0', 'ListNet', 'FairnessAtK', 0.73], ['TREC_58.0', 'Color-Blind', 'AP', 0.0], ['TREC_58.0', 'Color-Blind', 'NDCG@1', 0.07711762358266568], ['TREC_58.0', 'Color-Blind', 'NDCG@5', 0.16268736658638386], ['TREC_58.0', 'Color-Blind', 'NDCG@10', 0.18169393880276363], ['TREC_58.0', 'Color-Blind', 'rKL', 0.03161110549613829], ['TREC_58.0', 'Color-Blind', 'DTR', 0.682829941302739], ['TREC_58.0', 'Color-Blind', 'DIR', 1.567228463800318], ['TREC_58.0', 'Color-Blind', 'FairnessAtK', 0.73], ['TREC_58.0', 'FeldmanEtAl', 'AP', 0.0], ['TREC_58.0', 'FeldmanEtAl', 'NDCG@1', 0.8687065049013741], ['TREC_58.0', 'FeldmanEtAl', 'NDCG@5', 0.3309011398013807], ['TREC_58.0', 'FeldmanEtAl', 'NDCG@10', 0.23655695122812417], ['TREC_58.0', 'FeldmanEtAl', 'rKL', 0.08041066348997097], ['TREC_58.0', 'FeldmanEtAl', 'DTR', 0.6150962142129277], ['TREC_58.0', 'FeldmanEtAl', 'DIR', 1.542354484512419], ['TREC_58.0', 'FeldmanEtAl', 'FairnessAtK', 0.82], ['TREC_58.0', 'FOEIR-DPC', 'AP', 0.0], ['TREC_58.0', 'FOEIR-DPC', 'NDCG@1', 0.07711762358266568], ['TREC_58.0', 'FOEIR-DPC', 'NDCG@5', 0.2316239790476789], ['TREC_58.0', 'FOEIR-DPC', 'NDCG@10', 0.17197819792342162], ['TREC_58.0', 'FOEIR-DPC', 'rKL', 0.14800021951097234], ['TREC_58.0', 'FOEIR-DPC', 'DTR', 0.33398084501594527], ['TREC_58.0', 'FOEIR-DPC', 'DIR', 1.0577714535447047], ['TREC_58.0', 'FOEIR-DPC', 'FairnessAtK', 0.2], ['TREC_58.0', 'FOEIR-DTC', 'AP', 0.0], ['TREC_58.0', 'FOEIR-DTC', 'NDCG@1', 0.07711762358266568], ['TREC_58.0', 'FOEIR-DTC', 'NDCG@5', 0.2316239790476789], ['TREC_58.0', 'FOEIR-DTC', 'NDCG@10', 0.17197819792342162], ['TREC_58.0', 'FOEIR-DTC', 'rKL', 0.14800021951097234], ['TREC_58.0', 'FOEIR-DTC', 'DTR', 0.3169227545355864], ['TREC_58.0', 'FOEIR-DTC', 'DIR', 1.018135610924452], ['TREC_58.0', 'FOEIR-DTC', 'FairnessAtK', 0.2], ['TREC_58.0', 'FOEIR-DIC', 'AP', 0.0], ['TREC_58.0', 'FOEIR-DIC', 'NDCG@1', 0.07711762358266568], ['TREC_58.0', 'FOEIR-DIC', 'NDCG@5', 0.2316239790476789], ['TREC_58.0', 'FOEIR-DIC', 'NDCG@10', 0.17197819792342162], ['TREC_58.0', 'FOEIR-DIC', 'rKL', 0.14800021951097234], ['TREC_58.0', 'FOEIR-DIC', 'DTR', 0.3169227545355864], ['TREC_58.0', 'FOEIR-DIC', 'DIR', 1.018135610924452], ['TREC_58.0', 'FOEIR-DIC', 'FairnessAtK', 0.2], ['TREC_58.0', 'FAIR', 'AP', 0.0], ['TREC_58.0', 'FAIR', 'NDCG@1', 0.07711762358266568], ['TREC_58.0', 'FAIR', 'NDCG@5', 0.2316239790476789], ['TREC_58.0', 'FAIR', 'NDCG@10', 0.17197819792342162], ['TREC_58.0', 'FAIR', 'rKL', 8.40595079015932e-05], ['TREC_58.0', 'FAIR', 'DTR', 0.6828299394715814], ['TREC_58.0', 'FAIR', 'DIR', 1.5672284637970937], ['TREC_58.0', 'FAIR', 'FairnessAtK', 0.82], ['TREC_58.0', 'LFRanking', 'AP', 0.0], ['TREC_58.0', 'LFRanking', 'NDCG@1', 0.04677798551051927], ['TREC_58.0', 'LFRanking', 'NDCG@5', 0.030346348501057867], ['TREC_58.0', 'LFRanking', 'NDCG@10', 0.14765161953854494], ['TREC_58.0', 'LFRanking', 'rKL', 0.03161110549613829], ['TREC_58.0', 'LFRanking', 'DTR', 0.08629099708447766], ['TREC_58.0', 'LFRanking', 'DIR', 1.6955588367006664], ['TREC_58.0', 'LFRanking', 'FairnessAtK', 0.44], ['TREC_59.0', 'ListNet', 'AP', 0.0002777777777777778], ['TREC_59.0', 'ListNet', 'NDCG@1', 0.0006351759149535386], ['TREC_59.0', 'ListNet', 'NDCG@5', 0.007204288170681255], ['TREC_59.0', 'ListNet', 'NDCG@10', 0.07494646048963151], ['TREC_59.0', 'ListNet', 'rKL', 0.032643260249127524], ['TREC_59.0', 'ListNet', 'DTR', 0.14526903272532454], ['TREC_59.0', 'ListNet', 'DIR', 1.5157363918145557], ['TREC_59.0', 'ListNet', 'FairnessAtK', 1.0], ['TREC_59.0', 'Color-Blind', 'AP', 0.0002777777777777778], ['TREC_59.0', 'Color-Blind', 'NDCG@1', 0.0006351759149535386], ['TREC_59.0', 'Color-Blind', 'NDCG@5', 0.007204288170681255], ['TREC_59.0', 'Color-Blind', 'NDCG@10', 0.07494646048963151], ['TREC_59.0', 'Color-Blind', 'rKL', 0.032643260249127524], ['TREC_59.0', 'Color-Blind', 'DTR', 0.14526903272532454], ['TREC_59.0', 'Color-Blind', 'DIR', 1.5157363918145557], ['TREC_59.0', 'Color-Blind', 'FairnessAtK', 1.0], ['TREC_59.0', 'FeldmanEtAl', 'AP', 0.0002777777777777778], ['TREC_59.0', 'FeldmanEtAl', 'NDCG@1', 0.8580639703367409], ['TREC_59.0', 'FeldmanEtAl', 'NDCG@5', 0.29489335675695816], ['TREC_59.0', 'FeldmanEtAl', 'NDCG@10', 0.20390496752601886], ['TREC_59.0', 'FeldmanEtAl', 'rKL', 0.0775717567630102], ['TREC_59.0', 'FeldmanEtAl', 'DTR', 0.14526902718400578], ['TREC_59.0', 'FeldmanEtAl', 'DIR', 1.5157363918149536], ['TREC_59.0', 'FeldmanEtAl', 'FairnessAtK', 1.0], ['TREC_59.0', 'FOEIR-DPC', 'AP', 0.0006944444444444444], ['TREC_59.0', 'FOEIR-DPC', 'NDCG@1', 0.0006351759149535386], ['TREC_59.0', 'FOEIR-DPC', 'NDCG@5', 0.1873588798743902], ['TREC_59.0', 'FOEIR-DPC', 'NDCG@10', 0.13395491656134464], ['TREC_59.0', 'FOEIR-DPC', 'rKL', 0.09677718978102562], ['TREC_59.0', 'FOEIR-DPC', 'DTR', 0.062222539751693645], ['TREC_59.0', 'FOEIR-DPC', 'DIR', 0.8202645671528117], ['TREC_59.0', 'FOEIR-DPC', 'FairnessAtK', 0.2], ['TREC_59.0', 'FOEIR-DTC', 'AP', 0.0], ['TREC_59.0', 'FOEIR-DTC', 'NDCG@1', 0.0006351759149535386], ['TREC_59.0', 'FOEIR-DTC', 'NDCG@5', 0.1873588798743902], ['TREC_59.0', 'FOEIR-DTC', 'NDCG@10', 0.13395491656134464], ['TREC_59.0', 'FOEIR-DTC', 'rKL', 0.09677718978102562], ['TREC_59.0', 'FOEIR-DTC', 'DTR', 0.05839178263761245], ['TREC_59.0', 'FOEIR-DTC', 'DIR', 0.7970754444240417], ['TREC_59.0', 'FOEIR-DTC', 'FairnessAtK', 0.2], ['TREC_59.0', 'FOEIR-DIC', 'AP', 0.0], ['TREC_59.0', 'FOEIR-DIC', 'NDCG@1', 0.0006351759149535386], ['TREC_59.0', 'FOEIR-DIC', 'NDCG@5', 0.1873588798743902], ['TREC_59.0', 'FOEIR-DIC', 'NDCG@10', 0.13395491656134464], ['TREC_59.0', 'FOEIR-DIC', 'rKL', 0.09677718978102562], ['TREC_59.0', 'FOEIR-DIC', 'DTR', 0.058391782637612445], ['TREC_59.0', 'FOEIR-DIC', 'DIR', 0.7970754444240419], ['TREC_59.0', 'FOEIR-DIC', 'FairnessAtK', 0.2], ['TREC_59.0', 'FAIR', 'AP', 0.0002777777777777778], ['TREC_59.0', 'FAIR', 'NDCG@1', 0.0006351759149535386], ['TREC_59.0', 'FAIR', 'NDCG@5', 0.1873588798743902], ['TREC_59.0', 'FAIR', 'NDCG@10', 0.13395491656134464], ['TREC_59.0', 'FAIR', 'rKL', 0.0002212943664928874], ['TREC_59.0', 'FAIR', 'DTR', 0.14526902583752144], ['TREC_59.0', 'FAIR', 'DIR', 1.5157363918150366], ['TREC_59.0', 'FAIR', 'FairnessAtK', 1.0], ['TREC_59.0', 'LFRanking', 'AP', 0.0], ['TREC_59.0', 'LFRanking', 'NDCG@1', 0.002543122303796631], ['TREC_59.0', 'LFRanking', 'NDCG@5', 0.019061320475507012], ['TREC_59.0', 'LFRanking', 'NDCG@10', 0.019108815322021428], ['TREC_59.0', 'LFRanking', 'rKL', 0.032643260249127524], ['TREC_59.0', 'LFRanking', 'DTR', 0.0637087004307115], ['TREC_59.0', 'LFRanking', 'DIR', 1.685426077430345], ['TREC_59.0', 'LFRanking', 'FairnessAtK', 0.1], ['TREC_60.0', 'ListNet', 'AP', 0.0], ['TREC_60.0', 'ListNet', 'NDCG@1', 0.05344550183782405], ['TREC_60.0', 'ListNet', 'NDCG@5', 0.060504744460806885], ['TREC_60.0', 'ListNet', 'NDCG@10', 0.06659552711661057], ['TREC_60.0', 'ListNet', 'rKL', 0.0361513846472201], ['TREC_60.0', 'ListNet', 'DTR', 0.594255551971055], ['TREC_60.0', 'ListNet', 'DIR', 1.52241251400879], ['TREC_60.0', 'ListNet', 'FairnessAtK', 0.1], ['TREC_60.0', 'Color-Blind', 'AP', 0.0], ['TREC_60.0', 'Color-Blind', 'NDCG@1', 0.05344550183782405], ['TREC_60.0', 'Color-Blind', 'NDCG@5', 0.060504744460806885], ['TREC_60.0', 'Color-Blind', 'NDCG@10', 0.06659552711661057], ['TREC_60.0', 'Color-Blind', 'rKL', 0.0361513846472201], ['TREC_60.0', 'Color-Blind', 'DTR', 0.594255551971055], ['TREC_60.0', 'Color-Blind', 'DIR', 1.52241251400879], ['TREC_60.0', 'Color-Blind', 'FairnessAtK', 0.1], ['TREC_60.0', 'FeldmanEtAl', 'AP', 0.0], ['TREC_60.0', 'FeldmanEtAl', 'NDCG@1', 0.8604237314399243], ['TREC_60.0', 'FeldmanEtAl', 'NDCG@5', 0.329767489523204], ['TREC_60.0', 'FeldmanEtAl', 'NDCG@10', 0.2378883491681595], ['TREC_60.0', 'FeldmanEtAl', 'rKL', 0.08742440043615007], ['TREC_60.0', 'FeldmanEtAl', 'DTR', 0.5501029748671756], ['TREC_60.0', 'FeldmanEtAl', 'DIR', 1.5097341970078508], ['TREC_60.0', 'FeldmanEtAl', 'FairnessAtK', 0.9], ['TREC_60.0', 'FOEIR-DPC', 'AP', 0.0], ['TREC_60.0', 'FOEIR-DPC', 'NDCG@1', 0.05344550183782405], ['TREC_60.0', 'FOEIR-DPC', 'NDCG@5', 0.22856027285350733], ['TREC_60.0', 'FOEIR-DPC', 'NDCG@10', 0.1720541147042753], ['TREC_60.0', 'FOEIR-DPC', 'rKL', 0.10717769567592089], ['TREC_60.0', 'FOEIR-DPC', 'DTR', 0.25027790092104696], ['TREC_60.0', 'FOEIR-DPC', 'DIR', 0.9801528425445294], ['TREC_60.0', 'FOEIR-DPC', 'FairnessAtK', 0.2], ['TREC_60.0', 'FOEIR-DTC', 'AP', 0.0], ['TREC_60.0', 'FOEIR-DTC', 'NDCG@1', 0.05344550183782405], ['TREC_60.0', 'FOEIR-DTC', 'NDCG@5', 0.22856027285350733], ['TREC_60.0', 'FOEIR-DTC', 'NDCG@10', 0.1720541147042753], ['TREC_60.0', 'FOEIR-DTC', 'rKL', 0.10717769567592089], ['TREC_60.0', 'FOEIR-DTC', 'DTR', 0.22875544663227784], ['TREC_60.0', 'FOEIR-DTC', 'DIR', 0.8864006985797827], ['TREC_60.0', 'FOEIR-DTC', 'FairnessAtK', 0.2], ['TREC_60.0', 'FOEIR-DIC', 'AP', 0.0], ['TREC_60.0', 'FOEIR-DIC', 'NDCG@1', 0.05344550183782405], ['TREC_60.0', 'FOEIR-DIC', 'NDCG@5', 0.22856027285350733], ['TREC_60.0', 'FOEIR-DIC', 'NDCG@10', 0.1720541147042753], ['TREC_60.0', 'FOEIR-DIC', 'rKL', 0.10717769567592089], ['TREC_60.0', 'FOEIR-DIC', 'DTR', 0.22875139154937907], ['TREC_60.0', 'FOEIR-DIC', 'DIR', 0.8862354087651735], ['TREC_60.0', 'FOEIR-DIC', 'FairnessAtK', 0.2], ['TREC_60.0', 'FAIR', 'AP', 0.0], ['TREC_60.0', 'FAIR', 'NDCG@1', 0.05344550183782405], ['TREC_60.0', 'FAIR', 'NDCG@5', 0.22856027285350733], ['TREC_60.0', 'FAIR', 'NDCG@10', 0.1720541147042753], ['TREC_60.0', 'FAIR', 'rKL', 3.911878734717315e-05], ['TREC_60.0', 'FAIR', 'DTR', 0.5942555552120725], ['TREC_60.0', 'FAIR', 'DIR', 1.5224125140180291], ['TREC_60.0', 'FAIR', 'FairnessAtK', 0.9], ['TREC_60.0', 'LFRanking', 'AP', 0.0], ['TREC_60.0', 'LFRanking', 'NDCG@1', 0.0567951498160809], ['TREC_60.0', 'LFRanking', 'NDCG@5', 0.0500012720325504], ['TREC_60.0', 'LFRanking', 'NDCG@10', 0.039613524730632683], ['TREC_60.0', 'LFRanking', 'rKL', 0.0361513846472201], ['TREC_60.0', 'LFRanking', 'DTR', 0.12689297479637648], ['TREC_60.0', 'LFRanking', 'DIR', 1.8207453651525423], ['TREC_60.0', 'LFRanking', 'FairnessAtK', 0.1], ['GermanCreditAge25', 'Color-Blind', 'AP', 1.0], ['GermanCreditAge25', 'Color-Blind', 'NDCG@1', 1.0], ['GermanCreditAge25', 'Color-Blind', 'NDCG@5', 1.0], ['GermanCreditAge25', 'Color-Blind', 'NDCG@10', 1.0], ['GermanCreditAge25', 'Color-Blind', 'rKL', 0.0019415156418714987], ['GermanCreditAge25', 'Color-Blind', 'DTR', 1.0320457284696474], ['GermanCreditAge25', 'Color-Blind', 'DIR', 0.8082360087190942], ['GermanCreditAge25', 'Color-Blind', 'FairnessAtK', 0.21], ['GermanCreditAge25', 'FeldmanEtAl', 'AP', 0.0], ['GermanCreditAge25', 'FeldmanEtAl', 'NDCG@1', 0.8739393159214142], ['GermanCreditAge25', 'FeldmanEtAl', 'NDCG@5', 0.9754038016556424], ['GermanCreditAge25', 'FeldmanEtAl', 'NDCG@10', 0.9849148987659252], ['GermanCreditAge25', 'FeldmanEtAl', 'rKL', 0.0002189658221634689], ['GermanCreditAge25', 'FeldmanEtAl', 'DTR', 0.7319217913591702], ['GermanCreditAge25', 'FeldmanEtAl', 'DIR', 0.817111548400587], ['GermanCreditAge25', 'FeldmanEtAl', 'FairnessAtK', 0.4], ['GermanCreditAge25', 'FOEIR-DPC', 'AP', 0.0], ['GermanCreditAge25', 'FOEIR-DPC', 'NDCG@1', 1.0], ['GermanCreditAge25', 'FOEIR-DPC', 'NDCG@5', 0.992024198939576], ['GermanCreditAge25', 'FOEIR-DPC', 'NDCG@10', 0.996008774349139], ['GermanCreditAge25', 'FOEIR-DPC', 'rKL', 0.0005164197452750561], ['GermanCreditAge25', 'FOEIR-DPC', 'DTR', 1.0498838345130475], ['GermanCreditAge25', 'FOEIR-DPC', 'DIR', 1.1725261137391652], ['GermanCreditAge25', 'FOEIR-DPC', 'FairnessAtK', 0.275], ['GermanCreditAge25', 'FOEIR-DTC', 'AP', 0.0], ['GermanCreditAge25', 'FOEIR-DTC', 'NDCG@1', 1.0], ['GermanCreditAge25', 'FOEIR-DTC', 'NDCG@5', 0.992024198939576], ['GermanCreditAge25', 'FOEIR-DTC', 'NDCG@10', 0.996008774349139], ['GermanCreditAge25', 'FOEIR-DTC', 'rKL', 0.0005164197452750561], ['GermanCreditAge25', 'FOEIR-DTC', 'DTR', 0.7318120016666002], ['GermanCreditAge25', 'FOEIR-DTC', 'DIR', 0.8171115809857934], ['GermanCreditAge25', 'FOEIR-DTC', 'FairnessAtK', 0.275], ['GermanCreditAge25', 'FOEIR-DIC', 'AP', 0.0], ['GermanCreditAge25', 'FOEIR-DIC', 'NDCG@1', 1.0], ['GermanCreditAge25', 'FOEIR-DIC', 'NDCG@5', 0.992024198939576], ['GermanCreditAge25', 'FOEIR-DIC', 'NDCG@10', 0.996008774349139], ['GermanCreditAge25', 'FOEIR-DIC', 'rKL', 0.0005164197452750561], ['GermanCreditAge25', 'FOEIR-DIC', 'DTR', 0.7318120016666002], ['GermanCreditAge25', 'FOEIR-DIC', 'DIR', 0.8171115809857934], ['GermanCreditAge25', 'FOEIR-DIC', 'FairnessAtK', 0.275], ['GermanCreditAge25', 'FAIR', 'AP', 0.0], ['GermanCreditAge25', 'FAIR', 'NDCG@1', 1.0], ['GermanCreditAge25', 'FAIR', 'NDCG@5', 0.992024198939576], ['GermanCreditAge25', 'FAIR', 'NDCG@10', 0.996008774349139], ['GermanCreditAge25', 'FAIR', 'rKL', 0.005278158344446212], ['GermanCreditAge25', 'FAIR', 'DTR', 0.7319217805779846], ['GermanCreditAge25', 'FAIR', 'DIR', 0.8171115483999473], ['GermanCreditAge25', 'FAIR', 'FairnessAtK', 0.4], ['GermanCreditAge25', 'LFRanking', 'AP', 0.0], ['GermanCreditAge25', 'LFRanking', 'NDCG@1', 0.5900188277177677], ['GermanCreditAge25', 'LFRanking', 'NDCG@5', 0.5809408988215358], ['GermanCreditAge25', 'LFRanking', 'NDCG@10', 0.6184459420954967], ['GermanCreditAge25', 'LFRanking', 'rKL', 0.00017479126961586998], ['GermanCreditAge25', 'LFRanking', 'DTR', 2.414344055781246], ['GermanCreditAge25', 'LFRanking', 'DIR', 0.8620977846766792], ['GermanCreditAge25', 'LFRanking', 'FairnessAtK', 0.1], ['GermanCreditAge35', 'Color-Blind', 'AP', 1.0], ['GermanCreditAge35', 'Color-Blind', 'NDCG@1', 1.0], ['GermanCreditAge35', 'Color-Blind', 'NDCG@5', 1.0], ['GermanCreditAge35', 'Color-Blind', 'NDCG@10', 1.0], ['GermanCreditAge35', 'Color-Blind', 'rKL', 0.013221817354072955], ['GermanCreditAge35', 'Color-Blind', 'DTR', 1.019754037044476], ['GermanCreditAge35', 'Color-Blind', 'DIR', 0.9381180503122727], ['GermanCreditAge35', 'Color-Blind', 'FairnessAtK', 0.19], ['GermanCreditAge35', 'FeldmanEtAl', 'AP', 0.0], ['GermanCreditAge35', 'FeldmanEtAl', 'NDCG@1', 0.9258636625654516], ['GermanCreditAge35', 'FeldmanEtAl', 'NDCG@5', 0.9852813755214475], ['GermanCreditAge35', 'FeldmanEtAl', 'NDCG@10', 0.9887933280160903], ['GermanCreditAge35', 'FeldmanEtAl', 'rKL', 0.005706158481411188], ['GermanCreditAge35', 'FeldmanEtAl', 'DTR', 0.6984597308798866], ['GermanCreditAge35', 'FeldmanEtAl', 'DIR', 0.8317181132971276], ['GermanCreditAge35', 'FeldmanEtAl', 'FairnessAtK', 1.0], ['GermanCreditAge35', 'FOEIR-DPC', 'AP', 0.0], ['GermanCreditAge35', 'FOEIR-DPC', 'NDCG@1', 1.0], ['GermanCreditAge35', 'FOEIR-DPC', 'NDCG@5', 0.9950558376413722], ['GermanCreditAge35', 'FOEIR-DPC', 'NDCG@10', 0.9953176405700449], ['GermanCreditAge35', 'FOEIR-DPC', 'rKL', 0.0014587676946270854], ['GermanCreditAge35', 'FOEIR-DPC', 'DTR', 0.8952614762269684], ['GermanCreditAge35', 'FOEIR-DPC', 'DIR', 1.0647335023673499], ['GermanCreditAge35', 'FOEIR-DPC', 'FairnessAtK', 1.0], ['GermanCreditAge35', 'FOEIR-DTC', 'AP', 0.000625], ['GermanCreditAge35', 'FOEIR-DTC', 'NDCG@1', 1.0], ['GermanCreditAge35', 'FOEIR-DTC', 'NDCG@5', 0.9950558376413722], ['GermanCreditAge35', 'FOEIR-DTC', 'NDCG@10', 0.9953176405700449], ['GermanCreditAge35', 'FOEIR-DTC', 'rKL', 0.00020114630511698905], ['GermanCreditAge35', 'FOEIR-DTC', 'DTR', 0.6952573459933079], ['GermanCreditAge35', 'FOEIR-DTC', 'DIR', 0.8317181211204313], ['GermanCreditAge35', 'FOEIR-DTC', 'FairnessAtK', 1.0], ['GermanCreditAge35', 'FOEIR-DIC', 'AP', 0.0], ['GermanCreditAge35', 'FOEIR-DIC', 'NDCG@1', 1.0], ['GermanCreditAge35', 'FOEIR-DIC', 'NDCG@5', 0.9950558376413722], ['GermanCreditAge35', 'FOEIR-DIC', 'NDCG@10', 0.9953176405700449], ['GermanCreditAge35', 'FOEIR-DIC', 'rKL', 0.0024991583333635924], ['GermanCreditAge35', 'FOEIR-DIC', 'DTR', 0.6952573459933079], ['GermanCreditAge35', 'FOEIR-DIC', 'DIR', 0.8317181211204313], ['GermanCreditAge35', 'FOEIR-DIC', 'FairnessAtK', 1.0], ['GermanCreditAge35', 'FAIR', 'AP', 0.0], ['GermanCreditAge35', 'FAIR', 'NDCG@1', 1.0], ['GermanCreditAge35', 'FAIR', 'NDCG@5', 0.9950558376413722], ['GermanCreditAge35', 'FAIR', 'NDCG@10', 0.9953176405700449], ['GermanCreditAge35', 'FAIR', 'rKL', 0.010150743293173772], ['GermanCreditAge35', 'FAIR', 'DTR', 0.6984597334900676], ['GermanCreditAge35', 'FAIR', 'DIR', 0.8317181132970927], ['GermanCreditAge35', 'FAIR', 'FairnessAtK', 1.0], ['GermanCreditAge35', 'LFRanking', 'AP', 0.0006666666666666666], ['GermanCreditAge35', 'LFRanking', 'NDCG@1', 0.4244766672367305], ['GermanCreditAge35', 'LFRanking', 'NDCG@5', 0.5359489972650372], ['GermanCreditAge35', 'LFRanking', 'NDCG@10', 0.5701111603369171], ['GermanCreditAge35', 'LFRanking', 'rKL', 0.0003835098940015182], ['GermanCreditAge35', 'LFRanking', 'DTR', -0.001152059260052574], ['GermanCreditAge35', 'LFRanking', 'DIR', 0.9393771698471249], ['GermanCreditAge35', 'LFRanking', 'FairnessAtK', 1.0], ['GermanCreditSex', 'Color-Blind', 'AP', 1.0], ['GermanCreditSex', 'Color-Blind', 'NDCG@1', 1.0], ['GermanCreditSex', 'Color-Blind', 'NDCG@5', 1.0], ['GermanCreditSex', 'Color-Blind', 'NDCG@10', 1.0], ['GermanCreditSex', 'Color-Blind', 'rKL', 0.0008560666111084664], ['GermanCreditSex', 'Color-Blind', 'DTR', 1.0676765288142418], ['GermanCreditSex', 'Color-Blind', 'DIR', 1.2078624408835479], ['GermanCreditSex', 'Color-Blind', 'FairnessAtK', 1.0], ['GermanCreditSex', 'FeldmanEtAl', 'AP', 0.0], ['GermanCreditSex', 'FeldmanEtAl', 'NDCG@1', 1.0], ['GermanCreditSex', 'FeldmanEtAl', 'NDCG@5', 0.9938374232707657], ['GermanCreditSex', 'FeldmanEtAl', 'NDCG@10', 0.9969806736276445], ['GermanCreditSex', 'FeldmanEtAl', 'rKL', 0.003106296965339337], ['GermanCreditSex', 'FeldmanEtAl', 'DTR', 1.2053214485299488], ['GermanCreditSex', 'FeldmanEtAl', 'DIR', 1.200615018962351], ['GermanCreditSex', 'FeldmanEtAl', 'FairnessAtK', 1.0], ['GermanCreditSex', 'FOEIR-DPC', 'AP', 0.0], ['GermanCreditSex', 'FOEIR-DPC', 'NDCG@1', 1.0], ['GermanCreditSex', 'FOEIR-DPC', 'NDCG@5', 0.9938374232707657], ['GermanCreditSex', 'FOEIR-DPC', 'NDCG@10', 0.9969806736276445], ['GermanCreditSex', 'FOEIR-DPC', 'rKL', 0.0001326835290636454], ['GermanCreditSex', 'FOEIR-DPC', 'DTR', 1.0615793532536923], ['GermanCreditSex', 'FOEIR-DPC', 'DIR', 0.9996235612112241], ['GermanCreditSex', 'FOEIR-DPC', 'FairnessAtK', 1.0], ['GermanCreditSex', 'FOEIR-DTC', 'AP', 0.0], ['GermanCreditSex', 'FOEIR-DTC', 'NDCG@1', 1.0], ['GermanCreditSex', 'FOEIR-DTC', 'NDCG@5', 0.9938374232707657], ['GermanCreditSex', 'FOEIR-DTC', 'NDCG@10', 0.9969806736276445], ['GermanCreditSex', 'FOEIR-DTC', 'rKL', 0.0006443881077730051], ['GermanCreditSex', 'FOEIR-DTC', 'DTR', 1.0274927284200595], ['GermanCreditSex', 'FOEIR-DTC', 'DIR', 1.0324577827740509], ['GermanCreditSex', 'FOEIR-DTC', 'FairnessAtK', 1.0], ['GermanCreditSex', 'FOEIR-DIC', 'AP', 0.0], ['GermanCreditSex', 'FOEIR-DIC', 'NDCG@1', 1.0], ['GermanCreditSex', 'FOEIR-DIC', 'NDCG@5', 0.9938374232707657], ['GermanCreditSex', 'FOEIR-DIC', 'NDCG@10', 0.9969806736276445], ['GermanCreditSex', 'FOEIR-DIC', 'rKL', 0.0006011876301958505], ['GermanCreditSex', 'FOEIR-DIC', 'DTR', 1.0274927284200595], ['GermanCreditSex', 'FOEIR-DIC', 'DIR', 1.0324577827740509], ['GermanCreditSex', 'FOEIR-DIC', 'FairnessAtK', 1.0], ['GermanCreditSex', 'FAIR', 'AP', 0.0], ['GermanCreditSex', 'FAIR', 'NDCG@1', 1.0], ['GermanCreditSex', 'FAIR', 'NDCG@5', 0.9938374232707657], ['GermanCreditSex', 'FAIR', 'NDCG@10', 0.9969806736276445], ['GermanCreditSex', 'FAIR', 'rKL', 0.01722320454960484], ['GermanCreditSex', 'FAIR', 'DTR', 1.2053214485299488], ['GermanCreditSex', 'FAIR', 'DIR', 1.200615018962351], ['GermanCreditSex', 'FAIR', 'FairnessAtK', 1.0], ['GermanCreditSex', 'LFRanking', 'AP', 0.0006666666666666666], ['GermanCreditSex', 'LFRanking', 'NDCG@1', 0.4244766672367305], ['GermanCreditSex', 'LFRanking', 'NDCG@5', 0.5359489972650372], ['GermanCreditSex', 'LFRanking', 'NDCG@10', 0.5701111603369171], ['GermanCreditSex', 'LFRanking', 'rKL', 7.799398442239725e-05], ['GermanCreditSex', 'LFRanking', 'DTR', 8.392285020629542], ['GermanCreditSex', 'LFRanking', 'DIR', 1.1005969790938077], ['GermanCreditSex', 'LFRanking', 'FairnessAtK', 1.0], ['ProPublicaRace', 'Color-Blind', 'AP', 1.0], ['ProPublicaRace', 'Color-Blind', 'NDCG@1', 1.0], ['ProPublicaRace', 'Color-Blind', 'NDCG@5', 1.0], ['ProPublicaRace', 'Color-Blind', 'NDCG@10', 1.0], ['ProPublicaRace', 'Color-Blind', 'rKL', 0.016682325435550656], ['ProPublicaRace', 'Color-Blind', 'DTR', 0.8547870925705842], ['ProPublicaRace', 'Color-Blind', 'DIR', 0.9629659474893921], ['ProPublicaRace', 'Color-Blind', 'FairnessAtK', 0.11], ['ProPublicaRace', 'FeldmanEtAl', 'AP', 0.0], ['ProPublicaRace', 'FeldmanEtAl', 'NDCG@1', 0.9823432924734421], ['ProPublicaRace', 'FeldmanEtAl', 'NDCG@5', 0.9888615854144575], ['ProPublicaRace', 'FeldmanEtAl', 'NDCG@10', 0.9866676911619755], ['ProPublicaRace', 'FeldmanEtAl', 'rKL', 0.0062836440963618505], ['ProPublicaRace', 'FeldmanEtAl', 'DTR', 0.6300349517522434], ['ProPublicaRace', 'FeldmanEtAl', 'DIR', 0.7706536340794631], ['ProPublicaRace', 'FeldmanEtAl', 'FairnessAtK', 1.0], ['ProPublicaRace', 'FOEIR-DPC', 'AP', 0.2536674518292165], ['ProPublicaRace', 'FOEIR-DPC', 'NDCG@1', 1.0], ['ProPublicaRace', 'FOEIR-DPC', 'NDCG@5', 0.9910903775715563], ['ProPublicaRace', 'FOEIR-DPC', 'NDCG@10', 0.9881229015295414], ['ProPublicaRace', 'FOEIR-DPC', 'rKL', 0.04334863191491781], ['ProPublicaRace', 'FOEIR-DPC', 'DTR', 0.9493694869743436], ['ProPublicaRace', 'FOEIR-DPC', 'DIR', 1.2529038985176302], ['ProPublicaRace', 'FOEIR-DPC', 'FairnessAtK', 1.0], ['ProPublicaRace', 'FOEIR-DTC', 'AP', 0.43359594798752077], ['ProPublicaRace', 'FOEIR-DTC', 'NDCG@1', 1.0], ['ProPublicaRace', 'FOEIR-DTC', 'NDCG@5', 0.9910903775715563], ['ProPublicaRace', 'FOEIR-DTC', 'NDCG@10', 0.9881229015295414], ['ProPublicaRace', 'FOEIR-DTC', 'rKL', 0.042944665364539045], ['ProPublicaRace', 'FOEIR-DTC', 'DTR', 0.6463727080093745], ['ProPublicaRace', 'FOEIR-DTC', 'DIR', 0.7772646873513477], ['ProPublicaRace', 'FOEIR-DTC', 'FairnessAtK', 1.0], ['ProPublicaRace', 'FOEIR-DIC', 'AP', 0.43359594798752077], ['ProPublicaRace', 'FOEIR-DIC', 'NDCG@1', 1.0], ['ProPublicaRace', 'FOEIR-DIC', 'NDCG@5', 0.9910903775715563], ['ProPublicaRace', 'FOEIR-DIC', 'NDCG@10', 0.9881229015295414], ['ProPublicaRace', 'FOEIR-DIC', 'rKL', 0.042587974739221615], ['ProPublicaRace', 'FOEIR-DIC', 'DTR', 0.6463727080093745], ['ProPublicaRace', 'FOEIR-DIC', 'DIR', 0.7772646873513477], ['ProPublicaRace', 'FOEIR-DIC', 'FairnessAtK', 1.0], ['ProPublicaRace', 'FAIR', 'AP', 0.08984360410830998], ['ProPublicaRace', 'FAIR', 'NDCG@1', 1.0], ['ProPublicaRace', 'FAIR', 'NDCG@5', 0.9910903775715563], ['ProPublicaRace', 'FAIR', 'NDCG@10', 0.9881229015295414], ['ProPublicaRace', 'FAIR', 'rKL', 0.0009611442166659526], ['ProPublicaRace', 'FAIR', 'DTR', 0.6465053765304556], ['ProPublicaRace', 'FAIR', 'DIR', 0.7772654791076312], ['ProPublicaRace', 'FAIR', 'FairnessAtK', 1.0], ['ProPublicaSex', 'Color-Blind', 'AP', 1.0], ['ProPublicaSex', 'Color-Blind', 'NDCG@1', 1.0], ['ProPublicaSex', 'Color-Blind', 'NDCG@5', 1.0], ['ProPublicaSex', 'Color-Blind', 'NDCG@10', 1.0], ['ProPublicaSex', 'Color-Blind', 'rKL', 0.0019305697777996804], ['ProPublicaSex', 'Color-Blind', 'DTR', 0.9144847689990983], ['ProPublicaSex', 'Color-Blind', 'DIR', 0.7338268018223265], ['ProPublicaSex', 'Color-Blind', 'FairnessAtK', 0.1], ['ProPublicaSex', 'FeldmanEtAl', 'AP', 0.0], ['ProPublicaSex', 'FeldmanEtAl', 'NDCG@1', 0.947496140240558], ['ProPublicaSex', 'FeldmanEtAl', 'NDCG@5', 0.9852752621629928], ['ProPublicaSex', 'FeldmanEtAl', 'NDCG@10', 0.988597895135811], ['ProPublicaSex', 'FeldmanEtAl', 'rKL', 0.0107144829546421], ['ProPublicaSex', 'FeldmanEtAl', 'DTR', 0.6449639884132288], ['ProPublicaSex', 'FeldmanEtAl', 'DIR', 0.7272421299436819], ['ProPublicaSex', 'FeldmanEtAl', 'FairnessAtK', 0.73], ['ProPublicaSex', 'FOEIR-DPC', 'AP', 0.23704184704184703], ['ProPublicaSex', 'FOEIR-DPC', 'NDCG@1', 1.0], ['ProPublicaSex', 'FOEIR-DPC', 'NDCG@5', 0.9919027826490879], ['ProPublicaSex', 'FOEIR-DPC', 'NDCG@10', 0.9929250983215826], ['ProPublicaSex', 'FOEIR-DPC', 'rKL', 0.005432692100072185], ['ProPublicaSex', 'FOEIR-DPC', 'DTR', 1.0200353918068281], ['ProPublicaSex', 'FOEIR-DPC', 'DIR', 1.3034799929820733], ['ProPublicaSex', 'FOEIR-DPC', 'FairnessAtK', 0.275], ['ProPublicaSex', 'FOEIR-DTC', 'AP', 0.7863364784404935], ['ProPublicaSex', 'FOEIR-DTC', 'NDCG@1', 1.0], ['ProPublicaSex', 'FOEIR-DTC', 'NDCG@5', 0.9919027826490879], ['ProPublicaSex', 'FOEIR-DTC', 'NDCG@10', 0.9929250983215826], ['ProPublicaSex', 'FOEIR-DTC', 'rKL', 0.005620982030718697], ['ProPublicaSex', 'FOEIR-DTC', 'DTR', 0.6440296560017297], ['ProPublicaSex', 'FOEIR-DTC', 'DIR', 0.7272401281302407], ['ProPublicaSex', 'FOEIR-DTC', 'FairnessAtK', 0.275], ['ProPublicaSex', 'FOEIR-DIC', 'AP', 0.7863364784404935], ['ProPublicaSex', 'FOEIR-DIC', 'NDCG@1', 1.0], ['ProPublicaSex', 'FOEIR-DIC', 'NDCG@5', 0.9919027826490879], ['ProPublicaSex', 'FOEIR-DIC', 'NDCG@10', 0.9929250983215826], ['ProPublicaSex', 'FOEIR-DIC', 'rKL', 0.005620982030718697], ['ProPublicaSex', 'FOEIR-DIC', 'DTR', 0.6440296560017297], ['ProPublicaSex', 'FOEIR-DIC', 'DIR', 0.7272401281302407], ['ProPublicaSex', 'FOEIR-DIC', 'FairnessAtK', 0.275], ['ProPublicaSex', 'FAIR', 'AP', 0.0927198298777246], ['ProPublicaSex', 'FAIR', 'NDCG@1', 1.0], ['ProPublicaSex', 'FAIR', 'NDCG@5', 0.9919027826490879], ['ProPublicaSex', 'FAIR', 'NDCG@10', 0.9929250983215826], ['ProPublicaSex', 'FAIR', 'rKL', 3.798978614046747e-05], ['ProPublicaSex', 'FAIR', 'DTR', 0.6449639965316156], ['ProPublicaSex', 'FAIR', 'DIR', 0.7272421286864729], ['ProPublicaSex', 'FAIR', 'FairnessAtK', 0.73]]
    
    #fileNames = ['TREC_49.0','TREC_50.0','TREC_51.0','TREC_52.0','TREC_53.0','TREC_54.0','TREC_55.0','TREC_56.0','TREC_57.0','TREC_58.0','TREC_59.0','TREC_60.0','GermanCreditAge25','GermanCreditAge35','GermanCreditSex','ProPublicaSex','ProPublicaRace']
    
    print(results)
    
    finalResults = finalEval.calculateFinalEvaluation(results, fileNames)          
    
    print(finalResults)
    
    df = pd.DataFrame(np.array(finalResults).reshape(len(finalResults),4), columns = ['Data_Set_Name', 'Algorithm_Name', 'Measure', 'Value'])
    
    df.to_csv('results/evaluationResults.csv', index=(False))
    
    plotData()
   
    endTime = datetime.datetime.now()
    
    print("Total time of execution: "+str(endTime-startTime))
    
    
def evaluateLearning(ranking, dataSetName, queryNumbers, listNet = False, k = 100):
    """
    Evaluates the learning algorithms per query, creates an output file for each ranked query,
    and start the scoreBasedEval method for each query
    
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
        evalResults += (runMetrics(k, queryProtected, queryNonprotected, queryRanking, queryRanking, finalName, 'ListNet'))
            
        output.sort(key=lambda x: x[2], reverse=True)
        
        finalPrinting += output
        
        #only start scoreBasedEval if the algorithm is listNet (baseline)
        if listNet == True:
            #run the score based evaluation on the ranked candidate list
            evalResults += scoreBasedEval(finalName,"", k, True, queryProtected, queryNonprotected, queryRanking, listNet)
            
        try:     
            with open('rankings/ListNet/' + finalName +'.csv','w',newline='') as mf:
                writer = csv.writer(mf)
                writer.writerows(finalPrinting) 
        except Exception:
            raise Exception("Some error occured during file creation. Double check specifics.")
    
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
    
    #check if the given data comes from the base line algorithm ListNet
    #if it does not, construct candidates from the data
    if listNet == False:
        #creates Candidates from the preprocessed CSV files in folder preprocessedDataSets
        protected, nonProtected, originalRanking = cC.createScoreBased(dataSetPath)
    
    #creates a csv with candidates ranked with color-blind ranking
    createRankingCSV(originalRanking, 'Color-Blind/' + dataSetName + 'ranking.csv',k )
    #run the metrics ones for the color-blind ranking
    evalResults += (runMetrics(k, protected, nonProtected, originalRanking, originalRanking, dataSetName, 'Color-Blind'))
    
    
    #create ranking like Feldman et al.
    feldRanking, pathFeldman = feldmanRanking(protected, nonProtected, k, dataSetName)
    #Update the currentIndex of a candidate according to feldmanRanking
    feldRanking = updateCurrentIndex(feldRanking)
    #create CSV with rankings from FAIR
    createRankingCSV(feldRanking, pathFeldman,k)
    #evaluate FAIR with all available measures
    evalResults += (runMetrics(k, protected, nonProtected, feldRanking, originalRanking, dataSetName, 'FeldmanEtAl'))
    
    
    #run evaluations for FOEIR with different Fairness Constraints
    #run for FOEIR-DPC
    dpcRanking, dpcPath, isDPC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DPC', k)
    if isDPC == True:
        dpcRanking = updateCurrentIndex(dpcRanking)
        evalResults += (runMetrics(40, protected, nonProtected, dpcRanking, originalRanking, dataSetName, 'FOEIR-DPC'))
        createRankingCSV(dpcRanking, dpcPath,40)
        
    dtcRanking, dtcPath, isDTC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DTC', k)
    if isDTC == True:
        dtcRanking = updateCurrentIndex(dtcRanking)
        evalResults += (runMetrics(40, protected, nonProtected, dtcRanking, originalRanking, dataSetName, 'FOEIR-DTC'))
        createRankingCSV(dtcRanking, dtcPath,40)
        
    dicRanking, dicPath, isDIC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DIC', k)
    if isDIC == True:
        dicRanking = updateCurrentIndex(dicRanking)
        createRankingCSV(dicRanking, dicPath,40)
        evalResults += (runMetrics(40, protected, nonProtected, dicRanking, originalRanking, dataSetName, 'FOEIR-DIC'))
          
    #run evaluations for FAIR
    #run FAIR algorithm 
    FAIRRanking, notSelected, pathFAIR = runFAIR(dataSetName, protected, nonProtected, k)
    #Update the currentIndex of a candidate according to FAIR
    FAIRRanking = updateCurrentIndex(FAIRRanking)
    #create CSV with rankings from FAIR
    createRankingCSV(FAIRRanking, pathFAIR,k)
    #evaluate FAIR with all available measures
    evalResults += (runMetrics(k, protected, nonProtected, FAIRRanking, originalRanking, dataSetName, 'FAIR'))
    
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
            evalResults += (runMetrics(k, protected, nonProtected, LFRanking, originalRanking, dataSetName, 'LFRanking'))
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