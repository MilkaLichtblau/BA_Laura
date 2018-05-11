# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:22:43 2018

@author: Laura
"""

from __future__ import division
import scipy.optimize as optim
import LearningFairRankingOptimization
import utility
# a python script for optimization. Can be run from command line by following command
# runOptimization input_fn target_att sensi_value k acc_measure cut_point output_fn
# input_fn represents the csv file stores the source data
# target_att represents the target attribute to rank on 
# sensi_value is the value of sentitive attribute represents the protected group
# k represents the size of intermediate layer of neural network
# acc_measure is choose from ["scoreDiff", "positionDiff", "kendallDis", "spearmanDis", "pearsonDis"]
# cut_point is the cut position of ranking to compute split fairness measures.
# output_fn represents the output file of optimization results

# test of this script can be found in testOptimization.py

SCORE_DIVERGENCE="scoreDiff" # represent average score difference -ranking accuracy measure

def main(_csv_fn,_target_col,_sensi_bound,_k,_cut_point,_rez_fn):
    """
        Run the optimization process.
        Output evaluation results as csv file.
        Output results (accuracy, group fairness in op, values of group fairness measures) during optimization as txt files. 
        
        :param _csv_fn: The file name of input data stored in csv file
                        In csv file, one column represents one attribute of user
                        one row represents the feature vector of one user
        :param _target_col: The target attribute ranked on i.e. score of ranking
        :param _sensi_bound: The value of sensitve attribute to use as protected group, 0 or 1, usually 1 represnts belonging to protected group 
                             Applied for binary sensitve attribute
        :param _k: The number of clusters in the intermediate layer of neural network
        :param _cut_point: The cut off point of set-wise group fairness calculation
        :param _rez_fn: The file name to output optimization results
        :return: no returns.
    """        

    data,input_scores,pro_data,unpro_data,pro_index=utility.transformCSVdata(_csv_fn,_target_col,_sensi_bound)
    
    user_N = len(data)
    pro_N = len(pro_data)
    
    print("start opt")

    # initialize the optimization
    rez,bnd=LearningFairRankingOptimization.initOptimization(data,_k) 
    
    LearningFairRankingOptimization.lbfgsOptimize.iters=0                
    rez = optim.fmin_l_bfgs_b(LearningFairRankingOptimization.lbfgsOptimize, x0=rez, disp=1, epsilon=1e-5, 
                   args=(data, pro_data, unpro_data, input_scores, _k, 0.01,
                         1, 100, 0), bounds = bnd,approx_grad=True, factr=1e12, pgtol=1e-04,maxfun=15000, maxiter=15000)
    
    print ("End opt")
    # evaluation after converged
    estimate_scores,acc_value=LearningFairRankingOptimization.calculateEvaluateRez(rez,data,input_scores,_k)
    
    # prepare the result line to write
    # initialize the outputted csv file
    result_fn=_rez_fn+".csv"
    with open('..\results\\' + result_fn,'w') as mf:
        mf.write("UserN,pro_N,K,TargetAtt,AccMeasure,acc_value\n")
        rez_file=open(result_fn, 'a')
        
        rez_fline=str(user_N)+","+str(pro_N)+","+str(_k)+","+str(_target_col)+","+str(acc_value)+"\n"
        rez_file.write(rez_fline)
        rez_file.close()

    if __name__ == "__main__":
        main()
        
main("..\preprocessedDataSets\GermanCredit_age25pre.csv",0,1,4,10,"GermanCredit_age25_LFRankingOpt")