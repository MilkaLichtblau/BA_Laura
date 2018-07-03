# -*- coding: utf-8 -*-
"""
Created on Fri May 25 18:46:35 2018

@author: Laura
"""

import numpy as np 
from cvxopt import spmatrix, matrix, sparse, solvers
from src.csvProcessing.csvPrinting import createPCSV
from src.algorithms.FOEIR.Birkhoff import birkhoff_von_neumann_decomposition

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
    
    ranking.sort(key=lambda candidate: candidate.learnedScores, reverse=True)
    
    #set k to maximum default value
    if k > 50:
        k = 50
        print('Calculation of P and the Birkhoff decomposition for k larger than 50 will take a long time and consume a significant amount of memory. Therefore, we set k to a maximum of 50. If you like to change that default got to src/algorithms/FOEIR/runFOEIR.py and change row 33 accordingly.')
    
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
    
    #compute birkoff von neumann decomposition
    result = birkhoff_von_neumann_decomposition(x)
    
    theta = 0
    final = 0
    #choose permuation matrix with highest probability
    for coefficient, permutation_matrix in result:
        final += coefficient
        #print(coefficient)
        #print(permutation_matrix)
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
    
    for candidate in nRanking[:k]:
        candidate.qualifications = candidate.learnedScores
        
    return nRanking, True

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
    u = []
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX =[]
    
    for candidate in ranking[:k]:
        u.append(candidate.learnedScores)
    
    # initialize v with DCG
    v = np.arange(1,(k+1),1)
    v = 1/np.log2(1 + v + 1)
    v = np.reshape(v, (1,k))
    
    arrayU = np.asarray(u)
    
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
    
    #build statistical parity constraint
    f1 = initf.dot(v)
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
   
    try:
        sol = solvers.lp(c, G, h)
    except Exception:
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no unprotected items in data set.')
        return 0, False
    
    print('Finished solving LP with DIC.')
    
    return np.array(sol['x']), True

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
    u = []
    unproU = 0
    proU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX =[]
    
    for candidate in ranking[:k]:
        u.append(candidate.learnedScores)
    
    # initialize v with DCG
    v = np.arange(1,(k+1),1)
    v = 1/np.log2(1 + v + 1)
    v = np.reshape(v, (1,k))
    
    arrayU = np.asarray(u)
    
    #normalize input
    arrayU = (arrayU - np.min(arrayU))/(np.max(arrayU)-np.min(arrayU))
    
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
            proU += arrayU[i]
            
        else:
            
            unproCount += 1
            unproListX.append(i)
            unproU += arrayU[i]
            
    arrayU = np.reshape(arrayU, (k,1))
    
    uv = arrayU.dot(v)
    uv = uv.flatten()
    
    #negate objective function to convert maximization problem to minimization problem
    uv = np.negative(uv)
      
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
    
    f1 = initf.dot(v)
    
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
    try:
        sol = solvers.lp(c, G, h)
    except Exception:
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no unprotected items in data set.')
        return 0, False
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
    u = []
    unproU = 0
    proU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX =[]
    
    for candidate in ranking[:k]:
        u.append(candidate.learnedScores)
    
    # initialize v with DCG
    v = np.arange(1,(k+1),1)
    v = 1/np.log2(1 + v + 1)
    v = np.reshape(v, (1,k))
    
    arrayU = np.asarray(u)
    
    #normalize input
    arrayU = (arrayU - np.min(arrayU))/(np.max(arrayU)-np.min(arrayU))

    
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
            proU += arrayU[i]
            
        else:
            
            unproCount += 1
            unproListX.append(i)
            unproU += arrayU[i]
     
    arrayU = np.reshape(arrayU, (k,1))
    
    uv = arrayU.dot(v)
    uv = uv.flatten()
    
    #negate objective function to convert maximization problem to minimization problem
    uv = np.negative(uv)    
    
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
    
    f1 = initf.dot(v)
    
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
    try:
        sol = solvers.lp(c, G, h)
    except Exception:
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no unprotected items in data set.')
        return 0, False
    print('Finished solving LP with DTC.')
    
    return np.array(sol['x']), True
