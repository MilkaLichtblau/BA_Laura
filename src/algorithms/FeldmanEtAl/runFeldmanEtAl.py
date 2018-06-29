# -*- coding: utf-8 -*-

from scipy.stats.stats import percentileofscore
from scipy.stats.stats import scoreatpercentile

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

def feldmanRanking(protectedCandidates, nonProtectedCandidates, k, dataSetName):
    
    # ensure candidates are sorted by descending qualificiations
    nonProtectedCandidates.sort(key=lambda candidate: candidate.learnedScores, reverse=True)
    nonProtectedQualifications = [nonProtectedCandidates[i].learnedScores for i in range(len(nonProtectedCandidates))]
    
    protectedCandidates.sort(key=lambda candidate: candidate.learnedScores, reverse=True)
    protectedQualifications = [protectedCandidates[i].learnedScores for i in range(len(protectedCandidates))]

    ranking = []
    
    # create same distribution for protected and non-protected candidates
    for i, candidate in enumerate(protectedCandidates):
        if i >= k:
            # only need to adapt the scores for protected candidates up to required length
            # the rest will not be considered anyway
            break
        # find percentile of protected candidate
        p = percentileofscore(protectedQualifications, candidate.learnedScores)
        # find score of a non-protected in the same percentile
        score = scoreatpercentile(nonProtectedQualifications, p)
        candidate.qualification = score
        ranking.append(candidate)
    
    ranking += nonProtectedCandidates

    # create a colorblind ranking
    ranking.sort(key=lambda candidate: candidate.qualification, reverse=True)
    
    rankingResultsPath = "FeldmanEtAl/" + dataSetName + "ranking.csv"

    return ranking, rankingResultsPath

