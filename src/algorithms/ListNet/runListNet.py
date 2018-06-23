# -*- coding: utf-8 -*-
"""
Created on Sun May 27 19:36:19 2018

@author: Laura
"""

from src.algorithms.ListNet.listnet import ListNet

def runListNet(ranking, train, validate, test, k = 100, verb = 100, maxIter = 1000, val = 0.5):


    train_val_filename = train
    test_score_filename = validate
    test_noscore_filename = test
    verbose = verb
    max_iter = maxIter
    val_ratio = val
    rank = k

    agent = ListNet(verbose = verbose, max_iter = max_iter, val_ratio = val_ratio, n_thres_cand = rank)
    agent.fit(train_val_filename = train_val_filename)
    if test_score_filename:
        agent.test(ranking, filename = test_score_filename, noscore = False)
    if test_noscore_filename:
        ranking = agent.test(ranking, filename = test_noscore_filename, noscore = True)
        
    dataSetName = test.split('/')[-3]
    
    return ranking, dataSetName
   
        
        
        
        
                
     



            