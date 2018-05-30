# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:40:38 2018

@author: Laura
"""

import csv
import numpy as np

"""
Creates multiple CSV files each one storing a query from a raw CSV with 
multiple queries inside. 

We assume the same length for all queries
"""


class dataSetFormatForLearning:
    

    def createQueriesCSV(rawFilepath, outPath, queryLength, queryColumn, judgeColumn, sensiColumn, *scoreColumns):
            """
            
            @param rawFilepath:    Path of the raw input data csv file with multiple queries inside
            @param outPath:        outputpath of the folder where datasets are stored should be: 
                "../learningDataSets/nameOfDataSet/"
            @param queryLength:    The length of the given query, used to calculate normalized scores from 
            ranked positions in the judgeColumn
            @param queryColumn:    Column with query identifier inside
            @param scoreColumns:   list of columns with the learning scores inside
            @param judgeColumn:    Column with the relevance judgement inside. Here, we assume 
            that this column stores the position of the item in the ranking which we normalize to gain
            a score for judgement of each item.
            @param sensiColumn:    Column with the sensitive attribute, we assume a binary sensitive attr
            1 = protected, 0 = non-protected
            
            no return but creates a CSV files with name dataSet_queryNumber as follows:
                
                |Score 1|Score 2|...|Score N|Relevance rating as score|Group membership|
                
            """
            
            # error handling for parameters
            if not isinstance( rawFilepath, (str) ):
                raise TypeError("Input path must be a String.")
            if not isinstance( outPath, (str) ):
                raise TypeError("Output path of file must be a String.")
            if not isinstance( queryLength, (int) ):
                raise TypeError("Length of query must be an Integer value.")
            if not isinstance( queryColumn, (int) ):
                raise TypeError("Column for query must be an Integer value.")
            if not isinstance( judgeColumn, (int) ):
                raise TypeError("Column for judgement must be an Integer value.")
            if not isinstance( sensiColumn, (int) ):
                raise TypeError("Column for sensitive attribute must be an Integer value.")
            if not isinstance( scoreColumns, (tuple) ):
                raise TypeError("Columns for score values must be inputted as tuple.")
            if not isinstance( scoreColumns[0], (int) ):
                raise TypeError("Columns for score values must be Integer values.")
            
            #try to open csv file and save content in numpy array, if not found raise error
            try:
                with open(rawFilepath, newline='') as File:  
                    reader = csv.reader(File)
                    rWithHeader = [row for row in reader]
            except FileNotFoundError:
                raise FileNotFoundError("File could not be found. Please enter a valid path to a csv file.")
        
            #column not defined in input csv
            if queryColumn > len(rWithHeader[1]):
                raise IndexError("The requested query column number is not part of the input data set.")
            if judgeColumn > len(rWithHeader[1]):
                raise IndexError("The requested judgement column number is not part of the input data set.")
            if sensiColumn > len(rWithHeader[1]):
                raise IndexError("The requested column number for sensitive attribute is not part of the input data set.")
            
            #omit header
            dataSetWithoutHeader = rWithHeader[1:]
            
            dataSet = np.array(dataSetWithoutHeader)
            dataSet = dataSet[:,[queryColumn]]
            
            #print(dataSet)
            dataSet = dataSet.tolist()
            flat_list = [item for sublist in dataSet for item in sublist]
            queries = set(flat_list)
            queries = list(queries)
            
            output = []
            
            #extracting file name and creating output file path
            outFile = rawFilepath.split("/")
            outFile = outFile[-1].split(".")
            
            scoreColumns = list(scoreColumns)
            
            columns = []
            columns += scoreColumns
            columns += [judgeColumn, sensiColumn]
            
            #Create csv files for each query with above mentioned specifics
            for i in queries:
                output = []
                for row in dataSetWithoutHeader:
                    if i == row[queryColumn]:
                        #gain relevant columns, precisely, column with scores, column with label, and column with sensitive Attribute
                        row[judgeColumn] = float(row[judgeColumn]) / queryLength
                        output.append([row[index] for index in columns])
                
                #needs to be adjusted for other data sets
                outFilePath = outPath + outFile[0] +'_'+ i + '.csv'
                #write scores and labels to a csv file
                try:
                    with open(outFilePath, 'w', newline='') as csvOut:
                        writer = csv.writer(csvOut)
                        writer.writerows(output)
                except Exception:
                    raise Exception("Some error occured during file creation. Double check specifics")
                    
    
    # Call Method change variables according to data set only change folder name for output folder
    # Output folder needs to exist, otherwise an Exception will occur
    createQueriesCSV("../rawDataSets/W3C/W3C.csv","../learningDatasets/W3C/",1092, 0, 7, 1, 2,3,4,5,6)