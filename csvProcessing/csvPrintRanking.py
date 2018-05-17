# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:26:29 2018

@author: Laura
"""
import csv

def createRankingCSV(rankedCandidates, rankingResultsPath):
    
    ranking = []
        
    for i in range(len(rankedCandidates)):
    
        uid = str(rankedCandidates[i].uuid)
        originQ = str(rankedCandidates[i].originalQualification)
        quali = str(rankedCandidates[i].qualification)
        proAttr = str(rankedCandidates[i].isProtected)
            
        ranking.append([uid, originQ, quali, proAttr])
            
        
    with open('results/' + rankingResultsPath,'w',newline='') as mf:
             writer = csv.writer(mf)
             writer.writerows(ranking) 
        