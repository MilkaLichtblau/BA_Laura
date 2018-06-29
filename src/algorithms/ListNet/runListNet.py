# -*- coding: utf-8 -*-
"""
Created on Sun May 27 19:36:19 2018

@author: Laura
"""

from src.algorithms.ListNet.listnet import ListNet

def runListNet(ranking, train, validate, test, k = 100, verb = 50, maxIter = 1000, val = 0.5):
    
    """
    runs ListNet
    
    @param ranking: candidates for ranking
    @param train: path of training data
    @param validate: path of validation data
    @param test: path of test data
    @param k: length of ranking we want to produce
    @param verb: showing steps from the neural network
    @param maxIter: maximal iterations of the nueral network
    @param val: validation ratio
    
    return a list of candidates with learned scores from the ranking and the data set name
    
    """


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
   
        
        
        
        
                
     



            