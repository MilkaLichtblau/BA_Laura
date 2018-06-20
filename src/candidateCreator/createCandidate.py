"""
Created on Fri May 11 17:16:07 2018

@author: Laura
"""

import pandas as pd
import numpy as np
from src.candidateCreator.candidate import Candidate

"""

Create objects of type candidate to use for the ranking algorithms

part of code refers to https://github.com/MilkaLichtblau/FA-IR_Ranking

"""

class createCandidate():
    
    def createLearningCandidate(filename):
        """
        
        @param filename: Path of input file. Assuming preprocessed CSV file:
            
            sensitive_attribute | session | label as index value | feature_1 | ... | feature_n
            
            sensitive_attribute: is either 0 for non-protected or 1 for protected
            session: indicates the query identifier of the file
            score: we assume that score is given indirectly as enumeration, therefore we normalize 
            the score with 1 - score/len(query)
        
        return    a list with candidate objects from the inputed document, might contain multiple queries
                  
        """
        
        
        ranking = []
        queryRanking = []
        
        try:
            #with open(filename) as csvfile:
            data = pd.read_csv(filename)
        except FileNotFoundError:
            raise FileNotFoundError("File could not be found. Something must have gone wrong during preprocessing.") 
            
        queryNumber = data['session']

        queryNumber = queryNumber.drop_duplicates()
        
        for query in queryNumber:
            dataQuery = data.loc[data.session == query]
            #take the query length + 1 to make sure that we will not get a score == 0
            l = len(dataQuery)+1
            nonProtected = []
            protected = []
            for row in dataQuery.itertuples():
                features = np.asarray(row[4:])
                # access second row of .csv with protected attribute 0 = nonprotected group and 1 = protected group
                if row[1] == 0:
                    nonProtected.append(Candidate(1 - row[3]/l, 1- row[3]/l, [], row[3], row[2], features))
                else:
                    protected.append(Candidate(1 - row[3]/l, 1 - row[3]/l, "protectedGroup", row[3], row[2], features))
    
            queryRanking = nonProtected + protected
        
            # sort candidates by credit scores 
            protected.sort(key=lambda candidate: candidate.qualification, reverse=True)
            nonProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)
            
            #creating a color-blind ranking which is only based on scores
            queryRanking.sort(key=lambda candidate: candidate.qualification, reverse=True)
            
            ranking += queryRanking
            
        return ranking, queryNumber
    

    def createScoreBased(filename):
        """
        
        @param filename: Path of input file. Assuming CSV with sensitive 
        attribute in last column and score second last column. If param features
        is true, we further assume that features are in the remaining first columns
        @param 
        
        return    A list with protected candidates, a list with nonProtected candidates 
                  and a list with the whole colorblind ranking.
        """
        
        
        protected = []
        nonProtected = []
        ranking = []
        i = 0
        
        try:
            with open(filename) as csvfile:
                data = pd.read_csv(csvfile)
                for row in data.itertuples():
                    i += 1
                    features = np.asarray(row[:(-2)])
                    # access second row of .csv with protected attribute 0 = nonprotected group and 1 = protected group
                    if row[-1] == 0:
                        nonProtected.append(Candidate(row[-2], row[-2], [], i, [], features))
                    else:
                        protected.append(Candidate(row[-2], row[-2], "protectedGroup", i, [], features))
        except FileNotFoundError:
            raise FileNotFoundError("File could not be found. Something must have gone wrong during preprocessing.")                
    
        ranking = nonProtected + protected
    
        # sort candidates by credit scores 
        protected.sort(key=lambda candidate: candidate.qualification, reverse=True)
        nonProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)
        
        #creating a color-blind ranking which is only based on scores
        ranking.sort(key=lambda candidate: candidate.qualification, reverse=True)
    
        return protected, nonProtected, ranking
    