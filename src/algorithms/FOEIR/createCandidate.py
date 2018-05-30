"""
Created on Fri May 11 17:16:07 2018

@author: Laura
"""

import pandas as pd
from candidate import Candidate

"""

Create objects of type candidate to use for the ranking algorithms

"""

class createCandidate():
    
    def createLearningCandidate(filename, *columns):
        """
        
        @param filename: Path of input file. Assuming preprocessed CSV file with no header, 
                  columns for learning, one score column and a column with the sensitive attribute.
                  The first columns contain scores for learning. The second last column 
                  contains the true labels normalized as scores of [0,1], a higher score
                  indicating higher placement and the last column containing the group membership encoded in 0 for 
                  membership of the nonprotected group and in 1 for membership of the
                  protected group
        @param columns: columns for learning a ranking algorithm starting at index 1 rather than 0
        
        return    A list with protected candidates, a list with nonProtected candidates 
                  and a list with the whole colorblind ranking.
        """
        
        
        protected = []
        nonProtected = []
        ranking = []
        i = 0
        column = len(columns)+1
        
        try:
            with open(filename) as csvfile:
                data = pd.read_csv(csvfile, header=None)
                for row in data.itertuples():
                    i += 1
                    # access second row of .csv with protected attribute 0 = nonprotected group and 1 = protected group
                    if row[-1] == 0:
                        nonProtected.append(Candidate(row[:column], row[-2], [], i, []))
                    else:
                        protected.append(Candidate(row[:column], row[-2], "protectedGroup", i, []))
        except FileNotFoundError:
            raise FileNotFoundError("File could not be found. Something must have gone wrong during preprocessing.")                
    
        
        ranking = nonProtected + protected
    
        # sort candidates by credit scores 
        protected.sort(key=lambda candidate: candidate.qualification, reverse=True)
        nonProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)
        
        #creating a color-blind ranking which is only based on scores
        ranking.sort(key=lambda candidate: candidate.qualification, reverse=True)
    
        return protected, nonProtected, ranking
    

    def createScoreBased(filename):
        """
        
        @param filename: Path of input file. Assuming preprocessed CSV file with no header
                  and two columns. The first column containing the ranking scores
                  the second column containing the group membership encoded in 0 for 
                  membership of the nonprotected group and in 1 for membership of the
                  protected group
        
        return    A list with protected candidates, a list with nonProtected candidates 
                  and a list with the whole colorblind ranking.
        """
        
        
        protected = []
        nonProtected = []
        ranking = []
        i = 0
        
        try:
            with open(filename) as csvfile:
                data = pd.read_csv(csvfile, header=None)
                for row in data.itertuples():
                    i += 1
                    # access second row of .csv with protected attribute 0 = nonprotected group and 1 = protected group
                    if row[2] == 0:
                        nonProtected.append(Candidate(row[1], row[1], [], i, []))
                    else:
                        protected.append(Candidate(row[1], row[1], "protectedGroup", i, []))
        except FileNotFoundError:
            raise FileNotFoundError("File could not be found. Something must have gone wrong during preprocessing.")                
    
        ranking = nonProtected + protected
    
        # sort candidates by credit scores 
        protected.sort(key=lambda candidate: candidate.qualification, reverse=True)
        nonProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)
        
        #creating a color-blind ranking which is only based on scores
        ranking.sort(key=lambda candidate: candidate.qualification, reverse=True)
    
        return protected, nonProtected, ranking
    
    """
    pro, non, rank = create("../preprocessedDataSets/GermanCredit_age25pre.csv")
    
    for i in range(len(rank)):
        print(rank[i].qualification)
        print(rank[i].isProtected)
        print(rank[i].currentIndex)
    """
    