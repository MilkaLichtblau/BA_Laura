# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:42:05 2018

@author: Laura
"""

from post_processing_methods.fair_ranker.test import FairnessInRankingsTester
from scipy.stats.stats import percentileofscore
from scipy.stats.stats import scoreatpercentile
from utilsAndConstants.constants import ESSENTIALLY_ZERO

"""
Code taken from https://github.com/MilkaLichtblau/FA-IR_Ranking/blob/master/src/post_processing_methods/fair_ranker/create.py

    creates a ranking that promotes the protected candidates by adjusting the distribution of the
    qualifications of the protected and non-protected group

    IMPORTANT: THIS METHOD MODIFIES THE ORIGINAL LIST OF PROTECTED CANDIDATES!
    I.e. it modifies the qualification of the
    protected candidates. If the original list has to be preserved, it has to be deep-copied into a
    new data structure, before handed over into this method.

    steps:
        1. take a protected candidate x
        2. determine the percentile of that candidate within their group percentile(x)
        3. find a non-protected candidate y that has the same percentile(y) == percentile(x)
        4. assign the score of y to x
        5. goto 1

    Parameters:
    ----------
    :param protectedCandidates: array of protected candidates
    :param nonProtectedCandidates: array of non-protected candidates
    :param k: length of the ranking to return

    Return:
    ------
    a ranking of protected and non-protected candidates, which tries to have a better share of
    protected and non-protected candidates
    """

def feldmanRanking(protectedCandidates, nonProtectedCandidates, k):

    # ensure candidates are sorted by descending qualificiations
    protectedCandidates.sort(key=lambda candidate: candidate.qualification, reverse=True)
    nonProtectedCandidates.sort(key=lambda candidate: candidate.qualification, reverse=True)

    protectedQualifications = [protectedCandidates[i].qualification for i in range(len(protectedCandidates))]
    nonProtectedQualifications = [nonProtectedCandidates[i].qualification for i in range(len(nonProtectedCandidates))]


    # create same distribution for protected and non-protected candidates
    for i, candidate in enumerate(protectedCandidates):
        if i >= k:
            # only need to adapt the scores for protected candidates up to required length
            # the rest will not be considered anyway
            break
        # find percentile of protected candidate
        p = percentileofscore(protectedQualifications, candidate.qualification)
        # find score of a non-protected in the same percentile
        score = scoreatpercentile(nonProtectedQualifications, p)
        candidate.qualification = score

    # create a colorblind ranking
    return fairRanking(k, protectedCandidates, nonProtectedCandidates, ESSENTIALLY_ZERO, 0.1)

