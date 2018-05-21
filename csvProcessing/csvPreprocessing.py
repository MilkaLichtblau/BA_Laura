# -*- coding: utf-8 -*-
"""
Created on Fri May 11 17:16:07 2018

@author: Laura
"""

import csv
import numpy as np

"""
Creates a CSV file from an inputted CSV file which stores a ranking.
The ranking is based on the inputted score attribute which can be specified by column number 
starting at 0 like usual in CS and the output is ordered 
based on that score.
The last attribute of the inputted CSV file must be a label of 0 or 1 where a 1 denotes
membership of the protected group.
We assume that all input files possess a header and account for that in this method.
The results are then written to a file in preprocessedDataSets and named with the 
original file name + "pre.csv" for preprocessed. 
"""

class csvPreprocessing():

    def createScoreOrderedCSV(rawFilepath, column):
        """
        
        @param rawFilepath:    Path of the raw input data csv file
        @param column:         Specifies the column that stores the score on which the output file
                               should be ordered on. 
        
        return                 Path of CSV file with the preprocessed data set and order of data set
        """
        
        # error handling for parameters
        # column not an integer
        if not isinstance( column, (int) ):
            raise TypeError("Column must be an Integer value.")
        
        #try to open csv file and save content in numpy array, if not found raise error
        try:
            with open(rawFilepath, newline='') as File:  
                reader = csv.reader(File)
                rWithHeader = np.array([row for row in reader])
        except FileNotFoundError:
            raise FileNotFoundError("File could not be found. Please enter a valid path to a csv file.")
    
        #column not defined in input csv
        if column >= len(rWithHeader[0]):
            raise IndexError("The requested score column number is not part of the input data set.")
            
        #last column should be column with label for sensitive attribute
        if column == len(rWithHeader[0])-1:
            raise IndexError("The requested score column number should be the label column with the indication of group membership. Please check input specifications in description for further information.")
        
        #omit header
        helper = rWithHeader[1:]
        
        #gain relevant columns, precisely, column with score and column with label
        helper = helper[:,[column,-1]]
        
        #convert string values into float to sort ranking
        ranking = np.array([(float(row[0]),float(row[1])) for row in helper])
        
        #sort ranking descending
        ranking = ranking[(-ranking[:,0]).argsort()]
        
        #extracting file name and creating output file path
        outFile = rawFilepath.split("/")
        outFile = outFile[-1].split(".")
        outFilePath = "preprocessedDataSets/" + outFile[0] + "pre.csv"
        
        #write scores and labels to a csv file
        with open(outFilePath, 'w', newline='') as csvOut:
            writer = csv.writer(csvOut)
            writer.writerows(ranking)
            
        return outFilePath



#nameOut = createScoreOrderedCSV("../dataSets/GermanCredit/GermanCredit_age25.csv", 2)

#print(nameOut)