# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:43:39 2018

@author: Laura
"""

import pandas as pd


"""

***IMPORTANT INFORMATION***

Creates multiple CSV files each one storing a multiple queries from a raw CSV with 
multiple queries inside. 

We assume the same length for all queries

We create 5 fold cross-validation sets so folder structure for that needs to be
present beforehand:
        ../../learningDataSet/nameOfDataSet/
        /fold1
        /fold2
        /fold3
        /fold4
        /fold5
            
"""

def createQueriesCSV(rawFilepath, outPath, queryColumn, judgeColumn, sensiColumn, *scoreColumns):
    
    """
            
    @param rawFilepath:    Path of the raw input data csv file with multiple queries inside
    @param outPath:        outputpath of the folder where datasets are stored should be: 
                            "../../learningDataSets/nameOfDataSet/"
    @param queryColumn:    Column with query identifier inside
    @param scoreColumns:   list of columns with the learning scores inside
    @param judgeColumn:    Column with the relevance judgement inside. Here, we assume 
    that this column stores the position of the item in the ranking which we normalize to gain
    a score for judgement of each item.
    @param sensiColumn:    Column with the sensitive attribute, we assume a binary sensitive attr
    1 = protected, 0 = non-protected
    
            no return but creates CSV files with name test.csv, train.csv, and validation.csv in the following format:
                
                |sensitive_attribute|session|score|feature_1|feature_2|...|feature_N|
                
            and devides queries in five fold cross validation each fold having a train (80%), validation(10%) and test (10%) set

    sensitive_attribute denotes group membership and is either 0 or 1
    session is the query id
    score has to be the ranking's order with regard to the query of whole numbers starting with 1 and ending with the end of the query
    feature_1 ... feature_N are created scores which are later used for rankings.
                
    """
    
    # error handling for parameters
    if not isinstance( rawFilepath, (str) ):
        raise TypeError("Input path must be a String.")
    if not isinstance( outPath, (str) ):
        raise TypeError("Output path of file must be a String.")
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
    
    #get data
    data = pd.read_csv(rawFilepath)
    
    header = list(data)
    
    #rename columns
    header[queryColumn] = 'session'
    header[judgeColumn] = 'score'
    header[sensiColumn] = 'sensitive_attribute'
    
    #initialize list for new order of columns
    order = ['sensitive_attribute','session','score']
    
    for i, column in enumerate(scoreColumns):
        i += 1
        header[column] = 'feature_'+str(i)
        order.append('feature_'+str(i))
      
    #rename columns
    data.columns = header
    
    queryNumber = data['session']

    queryNumber = queryNumber.drop_duplicates()
    
    total = len(queryNumber)
    
    #reorder columns
    data = data[order]
    
    #lists for creation of folds
    f1_tr = []
    f1_va = []
    f1_te = []
    f2_tr = []
    f2_va = []
    f2_te = []
    f3_tr = []
    f3_va = []
    f3_te = []
    f4_tr = []
    f4_va = []
    f4_te = []
    f5_tr = []
    f5_va = []
    f5_te = []
    
    #get queries for five fold cross validation
    for index, value in queryNumber.iteritems():
        
        per = value/total

        if per <= 0.20:
            
            f1_tr.append(value)
            f2_tr.append(value)
            f3_tr.append(value)
            f4_va.append(value)
            f5_te.append(value)
        
        elif per > 0.20 and per <= 0.40:
            
            f1_tr.append(value)
            f2_tr.append(value)
            f3_va.append(value)
            f4_te.append(value)
            f5_tr.append(value)
            
        elif per > 0.40 and per <= 0.60:
            
            f1_tr.append(value)
            f2_va.append(value)
            f3_te.append(value)
            f4_tr.append(value)
            f5_tr.append(value)
            
        elif per > 0.60 and per <= 0.80:
            
            f1_va.append(value)
            f2_te.append(value)
            f3_tr.append(value)
            f4_tr.append(value)
            f5_tr.append(value)
            
        elif per > 0.80:
            
            f1_te.append(value)
            f2_tr.append(value)
            f3_tr.append(value)
            f4_tr.append(value)
            f5_va.append(value)
            
    #print data to csv
    data.loc[data['session'].isin(f1_tr)].to_csv(outPath+'fold1/train.csv', index=False)
    data.loc[data['session'].isin(f1_va)].to_csv(outPath+'fold1/validation.csv', index=False)
    data.loc[data['session'].isin(f1_te)].to_csv(outPath+'fold1/test.csv', index=False)
    
    data.loc[data['session'].isin(f2_tr)].to_csv(outPath+'fold2/train.csv', index=False)
    data.loc[data['session'].isin(f2_va)].to_csv(outPath+'fold2/validation.csv', index=False)
    data.loc[data['session'].isin(f2_te)].to_csv(outPath+'fold2/test.csv', index=False)
    
    data.loc[data['session'].isin(f3_tr)].to_csv(outPath+'fold3/train.csv', index=False)
    data.loc[data['session'].isin(f3_va)].to_csv(outPath+'fold3/validation.csv', index=False)
    data.loc[data['session'].isin(f3_te)].to_csv(outPath+'fold3/test.csv', index=False)
    
    data.loc[data['session'].isin(f4_tr)].to_csv(outPath+'fold4/train.csv', index=False)
    data.loc[data['session'].isin(f4_va)].to_csv(outPath+'fold4/validation.csv', index=False)
    data.loc[data['session'].isin(f4_te)].to_csv(outPath+'fold4/test.csv', index=False)
    
    data.loc[data['session'].isin(f5_tr)].to_csv(outPath+'fold5/train.csv', index=False)
    data.loc[data['session'].isin(f5_va)].to_csv(outPath+'fold5/validation.csv', index=False)
    data.loc[data['session'].isin(f5_te)].to_csv(outPath+'fold5/test.csv', index=False)

    
    
createQueriesCSV("../../rawDataSets/TREC/TREC.csv","../../learningDataSets/TREC/", 0, 7, 1, 2,3,4,5,6)