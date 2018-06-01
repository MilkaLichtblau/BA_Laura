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
import os
import csv
#from createCandidate import createCandidate as cC


def runFOEIR(ranking, dataSetName, algoName, k = 40):
    
    """
    
    """
    
    needToCreateCSVWithP = True
    
    if k > 40:
        k = 40
        print('Calculation of P for k larger than 40 will not yield any results but just crash the program. Therefore k will be set to 40.')
    
    if algoName == 'FOEIR-DIC':
        filepath = 'doublyStochasticPropMatrix/FOEIR-DIC/' + dataSetName + str(k) + '.csv'
        
        if os.path.exists(filepath) == True and '_' not in filepath:
            x = readPFromFile(filepath, algoName)
            needToCreateCSVWithP = False
        else:
            x = solveLPWithDIC(ranking, k)
    elif algoName == 'FOEIR-DPC':
    
        filepath = 'doublyStochasticPropMatrix/FOEIR-DPC/' + dataSetName + str(k) + '.csv'
        
        if os.path.exists(filepath) == True and '_' not in filepath:
            x = readPFromFile(filepath, algoName)
            needToCreateCSVWithP = False
        else:
            x = solveLPWithDIC(ranking, k)
    elif algoName == 'FOEIR-DTC':
        
        filepath = 'doublyStochasticPropMatrix/FOEIR-DTC/' + dataSetName + str(k) + '.csv'
        
        if os.path.exists(filepath) == True and '_' not in filepath:
            x = readPFromFile(filepath, algoName)
            needToCreateCSVWithP = False
        else:
            x = solveLPWithDIC(ranking, k)
        
    x = np.reshape(x,(k,k))

    x = np.asarray(x, dtype='float64')

    if needToCreateCSVWithP == True:
        createPCSV(x, dataSetName, algoName, k)
    
    newRanking = createRanking(x, ranking, k)
    
    rankingResultsPath = algoName + '/' + dataSetName + "ranking.csv"
    
    return newRanking, rankingResultsPath
    
    
def createRanking(x, nRanking, k):
    
    # Create a doubly stochastic matrix.
    #
    # D = numpy.array(...)
    
    # The decomposition is given as a list of pairs in which the right element
    # is a permutation matrix and the left element is the scalar coefficient
    # applied to that permutation matrix in the convex combination
    # representation of the doubly stochastic matrix.
    
    #round x to 0 decimal points because otherwise implementation of birkhoff_von_neumann_decomposition will not work.    
    x = np.around(x, 0)
    
    result = birkhoff_von_neumann_decomposition(x)
    
    eta = 0
    
    for coefficient, permutation_matrix in result:
        if eta < coefficient:
            eta = coefficient
            ranking = permutation_matrix
            
    positions = np.nonzero(ranking)[1]

    positions = positions.tolist()
    
    for p, candidate in zip(positions,nRanking[:k]):
        candidate.currentIndex = p+1
    
    #sort candidates according to new index
    nRanking.sort(key=lambda candidate: candidate.currentIndex, reverse=False)
    
    return nRanking

def solveLPWithDIC(ranking, k):
    
    print('Start building LP with DIC.')    
    #calculate the attention vector v using 1/log(1+indexOfRanking)
    v = []  
    u = []
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX =[]
    
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
    
    return np.array(sol['x'], dtype=np.float)

def solveLPWithDPC(ranking, k):
    
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
            proU += ranking[i].originalQualification
            
        else:
            
            unproCount += 1
            unproListX.append(i)
            unproU += ranking[i].originalQualification
            
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
    
    return np.array(sol['x'])

def solveLPWithDTC(ranking, k):
    
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
            proU += ranking[i].originalQualification
            
        else:
            
            unproCount += 1
            unproListX.append(i)
            unproU += ranking[i].originalQualification
            
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