# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:26:29 2018

@author: Laura
"""
import csv

def createRankingCSV(rankedCandidates, rankingResultsPath, k):
    
    """
    Prints the ranking output to a CSV file.
    
    @param rankedCandidates: list with ranked candidates
    @param rankingResultsPath: the path for the rankings should look like this
    algorithmName/dataSetName.csv
    
    no return
    """
    
    ranking = []
        
    for i in range(k):
    
        uid = str(rankedCandidates[i].uuid)
        originQ = str(rankedCandidates[i].originalQualification)
        quali = str(rankedCandidates[i].qualification)
        proAttr = str(rankedCandidates[i].isProtected)
            
        ranking.append([uid, originQ, quali, proAttr])
            
    try:     
        with open('rankings/' + rankingResultsPath,'w',newline='') as mf:
            writer = csv.writer(mf)
            writer.writerows(ranking) 
    except Exception:
            raise Exception("Some error occured during file creation. Double check specifics.")