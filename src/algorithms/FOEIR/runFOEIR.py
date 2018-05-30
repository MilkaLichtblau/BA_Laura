# -*- coding: utf-8 -*-
"""
Created on Fri May 25 18:46:35 2018

@author: Laura
"""

import numpy as np 
import math
from cvxopt import spmatrix, matrix, sparse, solvers

from createCandidate import createCandidate as cC

def runFOEIR(ranking, protected, unprotected):
    
    #solveLPWithoutFairness(ranking,protected,unprotected)
    
    k = len(ranking)
    
    print('Start setting up LP.')    
    #calculate the attention vector v using 1/log(1+indexOfRanking)
    v = []  
    u = []
    for candidate in ranking:
        u.append(candidate.originalQualification)
        v.append(1 / math.log((1 + candidate.originalIndex),2))
    
    arrayU = np.asarray(u)
    arrayV = np.asarray(v)
    
    arrayU = np.reshape(arrayU, (k,1))
    arrayV = np.reshape(arrayV, (1,k))
    
    uv = arrayU.dot(arrayV)
    uv = uv.flatten()
    print(uv)
    
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
         
    
    print('Start setting up matrices.')
    
    A = spmatrix(1.0, range(k**2), range(k**2))
    print('SetupA')
    A1 = spmatrix(-1.0, range(k**2), range(k**2))
    print('SetupA1')
    M = spmatrix(1.0, I,J)
    print('SetupM')
    M1 = spmatrix(1.0, I2,J)
    print('SetupM1')
    h1 = matrix(1.0, (k,1))
    print('Setuph1')
    b = matrix(1.0, (k**2,1))
    print('Setupb')
    d = matrix(0.0, (k**2,1))
    print('Setupd')
    c = matrix(uv)
    #print (np.array(c))
    print('LP setup finished.')
    
    #G = matrix([M,M1,A,A1])
    #print('setup G')
    G = sparse([M,M1,A,A1])
    print(len(G))
    print('making G sparse')
    h = matrix([h1,h1,b,d])
    print('setup h')
    
    sol = solvers.lp(c, G, h)
    
    print(np.array(sol['x']))
    
    

    """
    P = variable(k**2)
    y = variable()
    c1 = ( 2*x+y <= 3 )
    c2 = ( x+2*y <= 3 )
    c3 = ( x >= 0 )
    c4 = ( y >= 0 )
    lp1 = op(-4*x-5*y, [c1,c2,c3,c4])
    lp1.solve()
    lp1.status
    print(np.array(x.value))
    """
"""
def solveLPWithoutFairness(ranking,protected,unprotected):
    
    print('Start LP setup.')
    
    k = len(ranking)
    
    #define linear model to optimize
    model = pulp.LpProblem("Calculate probability matrix without constraints", pulp.LpMaximize)
    
    #create list with document indices
    listI = []
    for i in range(k):
        listI.append(i)
    
    #creat list with position indices
    listJ = []
    for j in range(k):
        listJ.append(j)
        
    #calculate the attention vector v using 1/log(1+indexOfRanking)
    v = []    
    for i in range(k):
        v.append(1 / math.log((1 + ranking[i].originalIndex),2))
        
    #create 
    uv = []
    for i in range(k):
        for j in range(k):
            uv.append(ranking[i].originalQualification * v[j])
        
        
    P = pulp.LpVariable.dicts("P",((i, j) for i in listI for j in listJ),lowBound=0, upBound=1)
    
    
    # Objective Function
    model += (
        pulp.lpSum([
            k * P[(i, j)]
            for i,k in zip(listI,v) for j in listJ])
    )
 
    #add constraints that ensure 1.T * P = 1.T, 1 denoting a column vector of size N
    for i in range(k):
        model += 1.0 * pulp.lpSum([P[i,j] for j in range(k)]) == 1.0
    
    #add constraints that ensure 1 * P = 1, 1 denoting a column vector of size N
    for j in range(k):
        model += 1.0 * pulp.lpSum([P[i,j] for i in range(k)]) == 1.0
        
    print('Start LP solving.')
    
    model.solve()
    print(pulp.LpStatus[model.status])

    print('Finished solving LP.')
    
    matrixP=[]
    for i in range(k):
        hlist = []
        for j in range(k):
            hlist.append(P[i,j].varValue)
        matrixP.append(hlist)

    print(matrixP)
"""
#creates Candidates from the preprocessed CSV files in folder preprocessedDataSets
protected, nonProtected, originalRanking = cC.createScoreBased("../../../preprocessedScoreSets/GermanCreditAge25pre.csv")

runFOEIR(originalRanking, protected, nonProtected)
