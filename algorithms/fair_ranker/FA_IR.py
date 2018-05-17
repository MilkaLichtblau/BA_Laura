# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:16:22 2018

@author: Laura
"""

from algorithms.fair_ranker.test import FairnessInRankingsTester

"""

Code taken from https://github.com/MilkaLichtblau/FA-IR_Ranking/blob/master/src/post_processing_methods/fair_ranker/create.py

"""


def fairRanking(k, protectedCandidates, nonProtectedCandidates, minProp, alpha):

    """
    creates a ranked output that satisfies the fairness definition in :class:'FairnessInRankingsTester'
    if k is larger than one of the candidate lists we have available, the ranking is filled up with
    candidates from the other group, i.e. if all protected candidates already appear in the ranking
    the left over positions are filled with non-protected

    Parameters:
    ----------
    k : int
        the expected length of the ranking

    protectedCandidates : [Candidates]
        array of protected class:`candidates <datasetCreator.candidate.Candidate>`, assumed to be
        sorted by candidate qualification in descending order

    nonProtectedCandidates : [Candidates]
        array of non-protected class:`candidates <datasetCreator.candidate.Candidate>`, assumed to be
        sorted by candidate qualification in descending order

    minProp : float
        minimal proportion of protected candidates to appear in the fair ranking result

    alpha : float
        significance level for the binomial cumulative distribution function -> minimum probability at
        which a fair ranking contains the minProp amount of protected candidates

    Return:
    ------
    an array of class:`candidates <datasetCreator.candidate.Candidate>` that maximizes ordering and
    selection fairness

    the left-over candidates that were not selected into the ranking, sorted color-blindly
    """

    
    result = []
    gft = FairnessInRankingsTester(minProp, alpha, k, correctedAlpha=True)
    countProtected = 0

    idxProtected = 0
    idxNonProtected = 0

    for i in range(k):
        if idxProtected >= len(protectedCandidates) and idxNonProtected >= len(nonProtectedCandidates):
            # no more candidates available, return list shorter than k
            return result, []
        if idxProtected >= len(protectedCandidates):
            # no more protected candidates available, take non-protected instead
            result.append(nonProtectedCandidates[idxNonProtected])
            idxNonProtected += 1

        elif idxNonProtected >= len(nonProtectedCandidates):
            # no more non-protected candidates available, take protected instead
            result.append(protectedCandidates[idxProtected])
            idxProtected += 1
            countProtected += 1

        elif countProtected < gft.candidates_needed[i]:
            # add a protected candidate
            result.append(protectedCandidates[idxProtected])
            idxProtected += 1
            countProtected += 1

        else:
            # find the best candidate available
            if protectedCandidates[idxProtected].qualification >= nonProtectedCandidates[idxNonProtected].qualification:
                # the best is a protected one
                result.append(protectedCandidates[idxProtected])
                idxProtected += 1
                countProtected += 1
            else:
                # the best is a non-protected one
                result.append(nonProtectedCandidates[idxNonProtected])
                idxNonProtected += 1

    return result, __mergeTwoRankings(protectedCandidates[idxProtected:], nonProtectedCandidates[idxNonProtected:])


def __mergeTwoRankings(ranking1, ranking2):
    result = ranking1 + ranking2
    result.sort(key=lambda candidate: candidate.originalQualification, reverse=True)
    return result

