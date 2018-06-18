# -*- coding: utf-8 -*-
"""
Created on Fri May 25 18:46:35 2018

@author: Laura
"""

import numpy as np 
import math
from cvxopt import spmatrix, matrix, sparse, solvers
from src.csvProcessing.csvPrinting import createPCSV
from birkhoff import birkhoff_von_neumann_decomposition


def runFOEIR(ranking, dataSetName, algoName, k = 40):
    
    """
    Start the calculation of the ranking for FOEIR under a given fairness constraint
    either Disparate Impact (DI), Disparate Treatment (DT), or Demographic Parity (DP)
    
    @param ranking: List of candidate objects ordered color-blindly
    @param dataSetName: Name of the data set the candidates are from
    @param algoName: Name of inputed algorithm
    @param k: Length of the ranking to return, if longer than default, 
    set to default because otherwise computation will run out of memory
    
    return the new ranking and the path where to print the ranking to
    """
    
    #initialize as empty string
    rankingResultsPath = ''
    newRanking = ranking
    
    #set k to maximum default value
    if k > 40:
        k = 40
        print('Calculation of P for k larger than 40 will not yield any results but just crash the program. Therefore k will be set to 40.')
    
    #check for which constraint to comput the ranking
    if algoName == 'FOEIR-DIC':

        x, isRanked = solveLPWithDIC(ranking, k, dataSetName, algoName)
    elif algoName == 'FOEIR-DPC':
        
        x, isRanked = solveLPWithDIC(ranking, k, dataSetName, algoName)
    elif algoName == 'FOEIR-DTC':
        
        x, isRanked = solveLPWithDIC(ranking, k, dataSetName, algoName)
        
    if isRanked == True:
        
        x = np.reshape(x,(k,k))
    
        x = np.asarray(x, dtype='float64')
        
        #crate csv file with doubly stochastic matrix inside
        createPCSV(x, dataSetName, algoName, k)
        
        #creat the new ranking, if not possible, isRanked will be false and newRanking
        #will be equal to ranking
        newRanking, isRanked = createRanking(x, ranking, k, algoName, dataSetName)
        
        if isRanked == True:
            rankingResultsPath = algoName + '/' + dataSetName + "ranking.csv"
    
    return newRanking, rankingResultsPath, isRanked
    
    
def createRanking(x, nRanking, k, algoName, dataSetName):
    
    """
    Calculates the birkhoff-von-Neumann decomopsition using package available at
    https://github.com/jfinkels/birkhoff
    
    @param x: doubly stochastic matrix 
    @param nRanking: nRanking: List with candidate objects from the data set ordered color-blindly
    @param k: length of the ranking
    @param dataSetName: Name of the data set the candidates are from
    @param algoName: Name of inputed algorithm
    
    return the a list with candidate objects ordered according to the new ranking
    """
    
    #round x to 0 decimal points because otherwise implementation of birkhoff_von_neumann_decomposition will not work.    
    x = np.around(x, 0)
    
    #initialize one dimensional array with ones for condition testing
    h = np.ones((1,len(x)))
    
    #test if x is still a doubly stochastic matrix even after rounding
    a1 = np.sum(x, axis = 0) == h
    a2 = np.sum(x, axis = 1) == h.T
    
    #only compute brikhoff for doubly stochastic matrixes
    if np.all(a1) and np.all(a2):
    
        #compute birkoff von neumann decomposition
        result = birkhoff_von_neumann_decomposition(x)
    
        theta = 0
        
        #choose permuation matrix with highest probability
        for coefficient, permutation_matrix in result:
            if theta < coefficient:
                theta = coefficient
                ranking = permutation_matrix
        
        #get positions of each document        
        positions = np.nonzero(ranking)[1]
    
        #convert numpy array to iterable list
        positions = positions.tolist()
    
        #correct the index of the items in the ranking according to permutation matrix
        for p, candidate in zip(positions,nRanking[:k]):
            candidate.currentIndex = p+1
        
        #sort candidates according to new index
        nRanking.sort(key=lambda candidate: candidate.currentIndex, reverse=False)
        
        return nRanking, True

    #otherwise we cannot obtain a ranking, hence return the original list and false
    else:
        
        print('Cannot create a ranking for ' + algoName + ' on data set ' + dataSetName)
        
        return nRanking, False

def solveLPWithDIC(ranking, k, dataSetName, algoName):
    
    """
    Solve the linear program with DIC
    
    @param ranking: list of candidate objects in the ranking
    @param k: length of the ranking
    @param dataSetName: Name of the data set the candidates are from
    @param algoName: Name of inputed algorithm
    
    return doubly stochastic matrix as numpy array
    """
    
    print('Start building LP with DIC.')    
    #calculate the attention vector v using 1/log(1+indexOfRanking)
    v = []  
    u = []
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX =[]
    
    for candidate in ranking[:k]:
        u.append(candidate.learnedScores)
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
    #set up indices for column and row constraints
    for j in range(k**2):
        J.append(j)
    
    for i in range(k):
        for j in range(k):
            I.append(i)
            
    for i in range(k):
        for j in range(k):
            I2.append(j)
            
            
    for i in range(k):
        
        if ranking[i].isProtected == True:
            
            proCount += 1
            proListX.append(i)
            
        else:
            
            unproCount += 1
            unproListX.append(i)
        
    # check if there are protected items    
    if proCount == 0:
        
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no protected items in data set.')
        return 0, False
    # check if there are unprotected items
    if unproCount == 0:
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no unprotected items in data set.')
        return 0, False
    
    initf = np.zeros((k,1))
    
    initf[proListX] = 1/proCount
    initf[unproListX] = -(1/unproCount)
    
    f1 = initf.dot(arrayV)
    
    f1 = f1.flatten()
    f1 = np.reshape(f1, (1,k**2))
    
    f = matrix(f1)
         
    #set up constraints x <= 1
    A = spmatrix(1.0, range(k**2), range(k**2))
    #set up constraints x >= 0
    A1 = spmatrix(-1.0, range(k**2), range(k**2))
    #set up constraints that sum(rows)=1
    M = spmatrix(1.0, I,J)
    #set up constraints sum(columns)=1
    M1 = spmatrix(1.0, I2,J)
    #values for sums columns and rows == 1
    h1 = matrix(1.0, (k,1))
    #values for x<=1
    b = matrix(1.0, (k**2,1))
    #values for x >= 0
    d = matrix(0.0, (k**2,1))
    #construct objective function
    c = matrix(uv)
    #assemble constraint matrix as sparse matrix    
    G = sparse([M,M1,A,A1,f])
    #assemble constraint values
    
    h = matrix([h1,h1,b,d,0.0])
    
    print('Start solving LP with DIC.')
   
    sol = solvers.lp(c, G, h)
    
    print('Finished solving LP with DIC.')
    
    return np.array(sol['x'], dtype=np.float), True

def solveLPWithDPC(ranking, k, dataSetName, algoName):
    
    """
    Solve the linear program with DPC
    
    @param ranking: list of candidate objects in the ranking
    @param k: length of the ranking
    @param dataSetName: Name of the data set the candidates are from
    @param algoName: Name of inputed algorithm
    
    return doubly stochastic matrix as numpy array
    """
    
    print('Start building LP with DPC.')    
    #calculate the attention vector v using 1/log(1+indexOfRanking)
    v = []  
    u = []
    unproU = 0
    proU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX =[]
    
    for candidate in ranking[:k]:
        u.append(candidate.learnedScores)
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
    #set up indices for column and row constraints
    for j in range(k**2):
        J.append(j)
    
    for i in range(k):
        for j in range(k):
            I.append(i)
            
    for i in range(k):
        for j in range(k):
            I2.append(j)
            
            
    for i in range(k):
        
        if ranking[i].isProtected == True:
            
            proCount += 1
            proListX.append(i)
            proU += ranking[i].learnedScores
            
        else:
            
            unproCount += 1
            unproListX.append(i)
            unproU += ranking[i].learnedScores
      
    # check if there are protected items    
    if proCount == 0:
        
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no protected items in data set.')
        return 0, False
    # check if there are unprotected items
    if unproCount == 0:
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no unprotected items in data set.')
        return 0, False
        
    proU = proU / proCount
    unproU = unproU / unproCount          
    
    initf = np.zeros((k,1))
    
    initf[proListX] = 1/(proCount*proU)
    initf[unproListX] = -(1/(unproCount*unproU))
    
    f1 = initf.dot(arrayV)
    
    f1 = f1.flatten()
    f1 = np.reshape(f1, (1,k**2))
    
    f = matrix(f1)
         
    #set up constraints x <= 1
    A = spmatrix(1.0, range(k**2), range(k**2))
    #set up constraints x >= 0
    A1 = spmatrix(-1.0, range(k**2), range(k**2))
    #set up constraints that sum(rows)=1
    M = spmatrix(1.0, I,J)
    #set up constraints sum(columns)=1
    M1 = spmatrix(1.0, I2,J)
    #values for sums columns and rows == 1
    h1 = matrix(1.0, (k,1))
    #values for x<=1
    b = matrix(1.0, (k**2,1))
    #values for x >= 0
    d = matrix(0.0, (k**2,1))
    #construct objective function
    c = matrix(uv)
    #assemble constraint matrix as sparse matrix    
    G = sparse([M,M1,A,A1,f])
    
    #assemble constraint values
    h = matrix([h1,h1,b,d,0.0])
    
    print('Start solving LP with DPC.')
   
    sol = solvers.lp(c, G, h)
    
    print('Finished solving LP with DPC.')
    
    return np.array(sol['x']), True

def solveLPWithDTC(ranking, k, dataSetName, algoName):
    
    """
    Solve the linear program with DTC
    
    @param ranking: list of candidate objects in the ranking
    @param k: length of the ranking
    @param dataSetName: Name of the data set the candidates are from
    @param algoName: Name of inputed algorithm
    
    return doubly stochastic matrix as numpy array
    """
    
    print('Start building LP with DTC.')    
    #calculate the attention vector v using 1/log(1+indexOfRanking)
    v = []  
    u = []
    unproU = 0
    proU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX =[]
    
    for candidate in ranking[:k]:
        u.append(candidate.learnedScores)
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
    #set up indices for column and row constraints
    for j in range(k**2):
        J.append(j)
    
    for i in range(k):
        for j in range(k):
            I.append(i)
            
    for i in range(k):
        for j in range(k):
            I2.append(j)
            
            
    for i in range(k):
        
        if ranking[i].isProtected == True:
            
            proCount += 1
            proListX.append(i)
            proU += ranking[i].learnedScores
            
        else:
            
            unproCount += 1
            unproListX.append(i)
            unproU += ranking[i].learnedScores
     
    # check if there are protected items    
    if proCount == 0:
        
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no protected items in data set.')
        return 0, False
    # check if there are unprotected items
    if unproCount == 0:
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no unprotected items in data set.')
        return 0, False
        
    proU = proU / proCount
    unproU = unproU / unproCount          
    
    initf = np.zeros((k,1))
    
    initf[proListX] = (1/(proCount*proU))*arrayU[proListX]
    initf[unproListX] = (-(1/(unproCount*unproU))*arrayU[unproListX])
    
    f1 = initf.dot(arrayV)
    
    f1 = f1.flatten()
    f1 = np.reshape(f1, (1,k**2))
    
    f = matrix(f1)
         
    #set up constraints x <= 1
    A = spmatrix(1.0, range(k**2), range(k**2))
    #set up constraints x >= 0
    A1 = spmatrix(-1.0, range(k**2), range(k**2))
    #set up constraints that sum(rows)=1
    M = spmatrix(1.0, I,J)
    #set up constraints sum(columns)=1
    M1 = spmatrix(1.0, I2,J)
    #values for sums columns and rows == 1
    h1 = matrix(1.0, (k,1))
    #values for x<=1
    b = matrix(1.0, (k**2,1))
    #values for x >= 0
    d = matrix(0.0, (k**2,1))
    #construct objective function
    c = matrix(uv)
    #assemble constraint matrix as sparse matrix    
    G = sparse([M,M1,A,A1,f])
    
    #assemble constraint values
    h = matrix([h1,h1,b,d,0.0])
    
    print('Start solving LP with DTC.')
   
    sol = solvers.lp(c, G, h)
    
    print('Finished solving LP with DTC.')
    
    return np.array(sol['x']), True
