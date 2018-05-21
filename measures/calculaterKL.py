# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:25:34 2018

@author: Laura
"""

from __future__ import division
import numpy as np
import math
import measures.dataGenerator as dataGenerator

"""
Code taken from https://github.com/DataResponsibly/FairRank/blob/master/measures.py

Calculates the rKL and its normalizer value Z
Needs the data generator for calculation of Z
"""

# a python script define computation of fairness measures and accuracy measures
# test of this script can be found in testMeasures.py

LOG_BASE=2 # log base used in logorithm function

NORM_CUTPOINT=10 # cut-off point used in normalizer computation
NORM_ITERATION=10 # max iterations used in normalizer computation
NORM_FILE="measures/normalizer.txt" # externally text file for normalizers


def calculateNDFairness(_ranking,_protected_group, k,_normalizer):
    """
        Calculate group fairness value of the whole ranking.
        Calls function 'calculateFairness' in the calculation.
        :param _ranking: A permutation of N numbers (0..N-1) that represents a ranking of N individuals, 
                                e.g., [0, 3, 5, 2, 1, 4].  Each number is an identifier of an individual.
                                Stored as a python array.
        :param _protected_group: A set of identifiers from _ranking that represent members of the protected group
                                e.g., [0, 2, 3].  Stored as a python array for convenience, order does not matter.
        :param k: Cut range for the calculation of group fairness, e.g., 10, 20, 30,...
        :param _normalizer: The normalizer of the input _gf_measure that is computed externally for efficiency.
        :return: returns  fairness value of _ranking, a float, normalized to [0, 1]
    """
    
    # error handling for input type 
    if not isinstance( k, ( int ) ):
        raise TypeError("Input batch size must be an integer larger than 0")
    if not isinstance( _normalizer, (int, float, complex) ):
        raise TypeError("Input normalizer must be a number larger than 0")
    
    user_N=len(_ranking)
    pro_N=len(_protected_group)

    # error handling for input value
    if NORM_CUTPOINT > user_N:
        raise ValueError("Batch size should be less than input ranking's length")
    
    discounted_gf=0 #initialize the returned gf value
    for countni in range(user_N):
        countni=countni+1 #account for countni starting at 0
        if(countni%k == 0): #Calculate for multiple of k, e.g. 10, 20, 30, 40, 50
            ranking_cutpoint=_ranking[0:countni]
            pro_cutpoint=set(ranking_cutpoint).intersection(_protected_group)

            gf=calculateFairness(ranking_cutpoint,pro_cutpoint,user_N,pro_N)
            discounted_gf+=gf/math.log(countni+1,LOG_BASE) # log base -> global variable
            
            # make a call to compute, or look up, the normalizer; make sure to check that it's not 0!
            # generally, think about error handling

    if _normalizer==0:
        raise ValueError("Normalizer equals to zero")
    return discounted_gf/_normalizer



def calculateFairness(_ranking,_protected_group,_user_N,_pro_N):
    """
        Calculate the group fairness value of input ranking.
        Called by function 'calculateNDFairness'.
        :param _ranking: A permutation of N numbers (0..N-1) that represents a ranking of N individuals, 
                                e.g., [0, 3, 5, 2, 1, 4].  Each number is an identifier of an individual.
                                Stored as a python array.
                                Can be a total ranking of input data or a partial ranking of input data.
        :param _protected_group: A set of identifiers from _ranking that represent members of the protected group
                                e.g., [0, 2, 3].  Stored as a python array for convenience, order does not matter.
        :param _user_N: The size of input items 
        :param _pro_N: The size of input protected group
        :param _gf_measure: The group fairness measure to be used in calculation        
        :return: returns the value of selected group fairness measure of this input ranking
    """
      
    ranking_k=len(_ranking)
    pro_k=len(_protected_group)

    gf=calculaterKL(ranking_k,pro_k,_user_N,_pro_N)        
           

    return gf 


def calculaterKL(_ranking_k,_pro_k,_user_N,_pro_N):
    """
        Calculate the KL-divergence difference of input ranking        
        :param _ranking_k: A permutation of k numbers that represents a ranking of k individuals, 
                                e.g., [0, 3, 5, 2, 1, 4].  Each number is an identifier of an individual.
                                Stored as a python array.
                                Can be a total ranking of input data or a partial ranking of input data.
        :param _pro_k: A set of identifiers from _ranking_k that represent members of the protected group
                                e.g., [0, 2, 3].  Stored as a python array for convenience, order does not matter.
        :param _user_N: The size of input items 
        :param _pro_N: The size of input protected group                
        :return: returns the value of KL-divergence difference of this input ranking
    """
    px=_pro_k/(_ranking_k)
    qx=_pro_N/_user_N
    if px==0 or px ==1: # manually set the value of extreme case to avoid error of math.log function 
        px=0.001
    if qx == 0 or qx ==1:
        qx=0.001
        
    return (px*math.log(px/qx,LOG_BASE)+(1-px)*math.log((1-px)/(1-qx),LOG_BASE))

def getNormalizer(_user_N,_pro_N, dataSetName):
    """
        Retrieve the normalizer of the current setting in external normalizer dictionary.
        If not found, call function 'calculateNormalizer' to calculate the normalizer of input group fairness measure at current setting.
        Called separately from fairness computation for efficiency.
        :param _user_N: The total user number of input ranking
        :param _pro_N: The size of protected group in the input ranking
        
        :return: returns the maximum value of selected group fairness measure in _max_iter iterations
    """
    # read the normalizor dictionary that is computed externally for efficiency
    normalizer_dic=readNormalizerDictionary()

    # error handling for type  
    if not isinstance( _user_N, ( int) ):
        raise TypeError("Input user number must be an integer")
    if not isinstance( _pro_N, ( int) ):
        raise TypeError("Input size of protected group must be an integer")

    # error handling for value 
    if _user_N <=0:
        raise ValueError("Input a valud user number")
    if _pro_N <=0:
        raise ValueError("Input a valid protected group size")
    if _pro_N >= _user_N:
        raise ValueError("Input a valid protected group size")


    current_normalizer_key=str(_user_N)+","+str(_pro_N)+","+dataSetName
    if current_normalizer_key in normalizer_dic.keys():
        normalizer=normalizer_dic[current_normalizer_key]
    else:
        normalizer=calculateNormalizer(_user_N,_pro_N) 
        with open(NORM_FILE, 'a') as the_file:
            the_file.write(str(_user_N)+","+str(_pro_N)+","+dataSetName+":"+str(normalizer)+'\n')
          
    return float(normalizer)

def readNormalizerDictionary():
    """
        Retrieve recorded normalizer from external txt file that is computed external for efficiency.
        Normalizer file is a txt file that each row represents the normalizer of a combination of user number and protected group number.
        Has the format like this: user_N,pro_N,_gf_measure:normalizer
        Called by function 'getNormalizer'.
        :param : no parameter needed. The name of normalizer file is constant.     
        :return: returns normalizer dictionary computed externally.
    """
    try:
        with open(NORM_FILE) as f:
            lines = f.readlines()
    except EnvironmentError as e:
        print("Cannot find the normalizer txt file")
    
    
    normalizer_dic={}
    
    if not lines:
        return normalizer_dic
    
    for line in lines:
        normalizer=line.split(":")
        normalizer_dic[normalizer[0]]=normalizer[1]
    return normalizer_dic

def calculateNormalizer(_user_N,_pro_N):
    """
        Calculate the normalizer of input group fairness measure at input user and protected group setting.
        The function use two constant: NORM_ITERATION AND NORM_CUTPOINT to specify the max iteration and batch size used in the calculation.
        First, get the maximum value of input group fairness measure at different fairness probability.
        Run the above calculation NORM_ITERATION times.
        Then compute the average value of above results as the maximum value of each fairness probability.
        Finally, choose the maximum of value as the normalizer of this group fairness measure.
        
        :param _user_N: The total user number of input ranking
        :param _pro_N: The size of protected group in the input ranking 
        
        :return: returns the group fairness value for the unfair ranking generated at input setting
    """
    # set the range of fairness probability based on input group fairness measure
    f_probs=[0,0.98] 
    
    avg_maximums=[] #initialize the lists of average results of all iteration
    for fpi in f_probs:
        iter_results=[] #initialize the lists of results of all iteration
        for iteri in range(NORM_ITERATION):
            input_ranking=[x for x in range(_user_N)]
            protected_group=[x for x in range(_pro_N)]
            # generate unfair ranking using algorithm
            unfair_ranking=dataGenerator.generateUnfairRanking(input_ranking,protected_group,fpi)    
            # calculate the non-normalized group fairness value i.e. input normalized value as 1
            gf=calculateNDFairness(unfair_ranking,protected_group,NORM_CUTPOINT,1)
            iter_results.append(gf)
        avg_maximums.append(np.mean(iter_results))        
        
        
    return max(avg_maximums)