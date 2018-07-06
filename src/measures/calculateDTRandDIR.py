# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:16:56 2018

@author: Laura
"""

import numpy as np 
from cvxopt import spmatrix, matrix, sparse, solvers
from src.csvProcessing.csvPrinting import createPCSV
import csv

def calculatedTRandDIR(ranking, algoName, dataSetName, k = 40):
    
    """
    Calculates DTR and DIR
    
    @param ranking: list of candidates in the ranking
    @param algoName: Algorithm that produced the ranking
    @param dataSetName: Data set the ranking was produced for
    @param k: lenght of the given ranking
    
    returns a list with the list results for DIR and DTR together with the name of the data set, 
    the algorithm name and die name of the measure
    """
    
    results = []
    
    if k > 40:
        k = 40
        print('Calculation of P for k larger than 40 will not yield any results but just crash the program. Therefore k will be set to 40.')
       
    x = solveLPWithoutFairness(ranking, algoName, k)
    
    x = np.reshape(x,(k,k))
    
    x = np.asarray(x, dtype='float64')
    
    eval_DTR = dTR(ranking, k, x)
    eval_DIR = dIR(ranking, k, x)
    
    results.append([dataSetName, algoName, 'DTR', eval_DTR])
    results.append([dataSetName, algoName, 'DIR', eval_DIR])
    
    #print the doubly stochastic matrix to ensure varifiability
    createPCSV(x, dataSetName, algoName, k)
    
    return results

def dTR(ranking, k, x):
    
    """
    Calculate Disparate Treatment Ratio (DTR)
    
    @param ranking: list with candidates from a given ranking
    @param k: truncation point/length of the given ranking
    @param x: doubly Stochastic matrix for given ranking
    
    return DTR
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
    
    u = np.asarray(utility)
    #normalize input
    u = (u - np.min(u))/np.max(u)-np.min(u)
    
    # get normalized values for protected and non-protected group as well as other data needed
    for i in range(k):  
        if ranking[i].isProtected == True:
            #add relevances for positive group
            proU += u[i]
            #get number of protected elements in ranking
            proCount += 1
            #get index of protected elements in ranking
            proListX.append(i)
        else:
            #do the same as above for unprotected
            unproU += u[i]
            unproCount += 1
            unproListX.append(i)

    v = np.arange(1,(k+1),1)
    v = 1/np.log2(1 + v + 1)
    v = np.reshape(v, (1,k))
    
    proExposure = np.sum((np.sum((x[proListX]*v),axis=1)),axis=0)
    unproExposure = np.sum((np.sum((x[unproListX]*v),axis=1)),axis=0)

    #initialize penalties if one of the counters is zero
    top = 0
    bottom = 0.01

    #calculate value for each group
    if proCount != 0:
        proU = proU / proCount         
        proExposure = proExposure / proCount
        top = (proExposure / proU)
        
    if unproCount != 0:   
        unproU = unproU / unproCount 
        unproExposure = unproExposure / unproCount
        bottom = (unproExposure / unproU)
    
    #calculate DTR
    dTR = top / bottom
    
    return dTR

    
def dIR(ranking, k, x):
    
    """
    Calculate Disparate Impact Ratio (DIR)
    
    @param ranking: list with candidates from a given ranking
    @param k: truncation point/length of the given ranking
    @param x: doubly Stochastic matrix for given ranking
    
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
    
    u = np.asarray(utility)
    
    #normalize input
    u = (u - np.min(u))/(np.max(u)-np.min(u))
    
    # get normalized values for protected and non-protected group as well as other data needed
    for i in range(k):  
        if ranking[i].isProtected == True:
            #add relevances for positive group
            proU += u[i]
            #get number of protected elements in ranking
            proCount += 1
            #get index of protected elements in ranking
            proListX.append(i)
        else:
            #do the same as above for unprotected
            unproU += u[i]
            unproCount += 1
            unproListX.append(i)
    
    # initialize v with DCG
    v = np.arange(1,(k+1),1)
    v = 1/np.log2(1 + v + 1)
    v = np.reshape(v, (1,k))
    
    u = np.reshape(u, (k,1))
    
    # calculate CTR for each group
    proCTR = np.sum((np.sum((x[proListX]*u[proListX]*v),axis=1)),axis=0)
    unproCTR = np.sum((np.sum((x[unproListX]*u[unproListX]*v),axis=1)),axis=0)
    
    #initialize penalties if one of the counters is zero
    top = 0
    bottom = 0.01

    #calculate value for each group
    if proCount != 0:
        proU = proU / proCount
        proCTR = proCTR / proCount
        top = (proCTR / proU)
    
    if unproCount != 0:
        
        unproU = unproU / unproCount          
        
        unproCTR = unproCTR / unproCount
        
        bottom = (unproCTR / unproU)

    
    #calculate DIR
    dIR = top / bottom
    
    return dIR
    
    
def solveLPWithoutFairness(ranking,algoName, k):
    
    """
    Compute a doubly stochastic matrix with all probabilities for a given document to be 
    ranked at a given rank
    
    @param ranking: list of candidates in the ranking
    @param algoName: Algorithm that produced the ranking
    @param k: lenght of the given ranking
    
    return the doubly stochastic matrix
    """
    
    print('Start building LP without Fairness Constraints for ' +algoName)    
    #calculate the attention vector v using 1/log(1+indexOfRanking)
    v = []  
    u = []
    for candidate in ranking[:k]:
        u.append(candidate.originalQualification)
    
    arrayU = np.asarray(u)
    
    # initialize v with DCG
    v = np.arange(1,(k+1),1)
    v = 1/np.log2(1 + v + 1)
    v = np.reshape(v, (1,k))
    
    #normalize input
    arrayU = (arrayU - np.min(arrayU))/(np.max(arrayU)-np.min(arrayU))
    
    arrayU = np.reshape(arrayU, (k,1))
    
    
    uv = arrayU.dot(v)
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
    
    print('Start solving LP without Fairness Constraints for ' +algoName)
   
    sol = solvers.lp(c, G, h)
    
    print('Finished solving LP without Fairness Constraints.')
    
    return np.array(sol['x'])
