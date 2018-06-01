# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:16:56 2018

@author: Laura
"""

import numpy as np 
import math
from cvxopt import spmatrix, matrix, sparse, solvers
from src.csvProcessing.csvPrinting import createPCSV
import csv

def calculatedTRandDIR(ranking, algoName, dataSetName, k = 40):
    
    results = []
    
    if k > 40:
        k = 40
        print('Calculation of P for k larger than 40 will not yield any results but just crash the program. Therefore k will be set to 40.')
    
    if algoName == 'FOEIR-DIC':
        
        filepath = 'doublyStochasticPropMatrix/FOEIR-DIC/' + dataSetName + str(k) + '.csv'
        x = readPFromFile(filepath, algoName)
    elif algoName == 'FOEIR-DPC':
        
        filepath = 'doublyStochasticPropMatrix/FOEIR-DPC/' + dataSetName + str(k) + '.csv'
        x = readPFromFile(filepath, algoName)
        
    elif algoName == 'FOEIR-DTC':
        
        filepath = 'doublyStochasticPropMatrix/FOEIR-DTC/' + dataSetName + str(k) + '.csv'
        x = readPFromFile(filepath, algoName)
    else:    
        x = solveLPWithoutFairness(ranking, algoName, k)
    
    x = np.reshape(x,(k,k))
    
    x = np.asarray(x, dtype='float64')
    
    eval_DTR = dTR(ranking, k, x)
    eval_DIR = dIR(ranking, k, x)
    
    results.append([dataSetName, algoName, 'DTR', eval_DTR])
    results.append([dataSetName, algoName, 'DIR', eval_DIR])
    
    createPCSV(x, dataSetName, algoName, k)
    
    return results

def dTR(ranking, k, x):
    
    """
    Calculate Disparate Treatment Ratio (DTR)
    
    @param ranking: list with candidates from a given ranking
    @param k: truncation point/length of the given ranking
    
    return DTR
    """
    
    #initialize variables
    proU = 0
    unproU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX = []
    
    #calculate utility and exposure for protected and unprotected groups
    for i in range(k):
        
        if ranking[i].isProtected == True:
            
            proCount += 1
            proU += ranking[i].originalQualification
            proListX.append(i)
            
        else:
            
            unproCount += 1
            unproU += ranking[i].originalQualification
            unproListX.append(i)

    v = np.arange(1,(k+1),1)
    
    v = 1/np.log2(1 + v + 1)
    
    v = np.reshape(v, (1,k))
    
    proExposure = np.sum((np.sum((x[proListX]*v),axis=1)),axis=0)
    unproExposure = np.sum((np.sum((x[unproListX]*v),axis=1)),axis=0)

    #normalize with counter
    proU = proU / proCount
    unproU = unproU / unproCount          
    proExposure = proExposure / proCount
    unproExposure = unproExposure / unproCount
    
    #calculate DTR
    dTR = (proExposure / proU) / (unproExposure / unproU)
    
    return dTR

    
def dIR(ranking, k, x):
    
    """
    Calculate Disparate Impact Ratio (DIR)
    
    @param ranking: list with candidates from a given ranking
    @param k: truncation point/length of the given ranking
    
    return DIR
    """
    
    #initialize variables
    proU = 0
    unproU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX = []
    utility = []
    
    #calculate utility and click through rate (CTR) for protected and unprotected groups
    for i in range(k):
        #get relevances from ranking
        utility.append(ranking[i].originalQualification)
        
        if ranking[i].isProtected == True:
            #add relevances for positive group
            proU += ranking[i].originalQualification
            #get number of protected elements in ranking
            proCount += 1
            #get index of protected elements in ranking
            proListX.append(i)
        else:
            #do the same as above for unprotected
            unproU += ranking[i].originalQualification
            unproCount += 1
            unproListX.append(i)

    v = np.arange(1,(k+1),1)
    
    v = 1/np.log2(1 + v + 1)
    
    u = np.asarray(utility)
    u = np.reshape(u, (k,1))
    
    proCTR = np.sum((np.sum((x[proListX]*u[proListX]*v),axis=1)),axis=0)
    unproCTR = np.sum((np.sum((x[unproListX]*u[unproListX]*v),axis=1)),axis=0)

    #normalize with counter
    proU = proU / proCount
    unproU = unproU / unproCount          
    proCTR = proCTR / proCount
    unproCTR = unproCTR / unproCount
    
    #calculate DIR
    dIR = (proCTR / proU) / (unproCTR / unproU)
    
    return dIR
    
    
def solveLPWithoutFairness(ranking,algoName, k):
    
    print('Start building LP without Fairness Constraints for' +algoName)    
    #calculate the attention vector v using 1/log(1+indexOfRanking)
    v = []  
    u = []
    for candidate in ranking[:k]:
        u.append(candidate.originalQualification)
        v.append(1 / math.log((1 + candidate.originalIndex),2))
    
    arrayU = np.asarray(u)
    arrayV = np.asarray(v)
    
    arrayU = np.reshape(arrayU, (k,1))
    arrayV = np.reshape(arrayV, (1,k))
    
    uv = arrayU.dot(arrayV)
    uv = uv.flatten()
    
    #negate objective function to convert maximization problem to minimization problem
    uv = np.negative(uv)
    
    I = []
    J = []
    I2 = []
    
    for j in range(k**2):
        J.append(j)
    
    for i in range(k):
        for j in range(k):
            I.append(i)
            
    for i in range(k):
        for j in range(k):
            I2.append(j)
         
    
    A = spmatrix(1.0, range(k**2), range(k**2))
    
    A1 = spmatrix(-1.0, range(k**2), range(k**2))

    M = spmatrix(1.0, I,J)

    M1 = spmatrix(1.0, I2,J)

    h1 = matrix(1.0, (k,1))

    b = matrix(1.0, (k**2,1))

    d = matrix(0.0, (k**2,1))

    c = matrix(uv)
        
    G = sparse([M,M1,A,A1])
    h = matrix([h1,h1,b,d])
    
    print('Start solving LP without Fairness Constraints for' +algoName)
   
    sol = solvers.lp(c, G, h)
    
    print('Finished solving LP without Fairness Constraints.')
    
    return np.array(sol['x'])

def readPFromFile(filepath, algoName):
    
    #try to open csv file and save content in numpy array, if not found raise error
    try:
        with open(filepath, newline='') as File:  
            reader = csv.reader(File)
            x = np.array([row for row in reader])
    except FileNotFoundError:
        raise FileNotFoundError("Could not find file with P for" + algoName + '.')
        
    x = x.flatten()
    
    return x