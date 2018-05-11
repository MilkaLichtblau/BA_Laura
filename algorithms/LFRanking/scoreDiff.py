# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:25:09 2018

@author: Laura
"""

from __future__ import division
import numpy as np

# a python script define computation of fairness measures and accuracy measures
# test of this script can be found in testMeasures.py
# Code refers to https://github.com/DataResponsibly/FairRank/blob/master/measures.py

def calculateScoreDifference(_scores1,_scores2):
    """
        Calculate the average position-wise score difference
        between two sorted lists.  Lists are sorted in decreasing
        order of scores.  If lists are not sorted by descending- error.
        Only applied for two score lists with same size. 
        # check for no division by 0
        # check that each list is sorted in decreasing order of score
        :param _scores1: The first list of scores
        :param _scores2: The second list of scores         
        :return: returns the average score difference of two input score lists.
    """
    # error handling 
    if not isinstance(_scores1, (list, tuple, np.ndarray)) and not isinstance( _scores1 ):
        raise TypeError("First score list must be a list-wise structure defined by '[]' symbol")
    if not isinstance(_scores2, (list, tuple, np.ndarray)) and not isinstance( _scores2 ):
        raise TypeError("Second score list must be a list-wise structure defined by '[]' symbol")
    
    if len(_scores1)*len(_scores2) ==0:
        raise ValueError("Input score lists should have length larger than 0")
        
    if not descendingOrderCheck(_scores1):
        raise ValueError("First score list is not ordered by descending order")
    if not descendingOrderCheck(_scores2):
        raise ValueError("Second score list is not ordered by descending order")

    user_N=min(len(_scores1),len(_scores2)) # get the minimum user number of two score lists
    score_diff = 0
    for xi in range(user_N):        
        score_diff+=abs(_scores1[xi]-_scores2[xi])         
    score_diff=score_diff/user_N
    return score_diff

# Functions for error handling
def descendingOrderCheck(_ordered_list):
    """
        Check whether the input list is ordered descending. 
        
        :param _ordered_list: The input list that is ordered by descending order         
        :return: returns true if order of _ordered_list is descending else returns false.
    """       
    return all(earlier >= later for earlier, later in zip(_ordered_list, _ordered_list[1:]))