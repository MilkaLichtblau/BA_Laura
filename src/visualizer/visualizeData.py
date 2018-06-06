# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 21:24:57 2018

@author: Laura
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#constants for algorithms
COLORBLIND = 'Color-Blind'
ALGO_FAIR = 'FAIR'
ALGO_LFRANKING = 'LFRanking'
ALGO_FELDMAN = 'FeldmanEtAl'
ALGO_FOEIRDPC = 'FOEIR-DPC'
ALGO_FOEIRDTC = 'FOEIR-DTC'
ALGO_FOEIRDIC = 'FOEIR-DIC'

def plotNWN():

    algoList = [COLORBLIND, ALGO_FAIR,ALGO_LFRANKING,ALGO_FELDMAN, ALGO_FOEIRDPC, ALGO_FOEIRDTC, ALGO_FOEIRDIC]
    
    #x = pd.read_csv('C:/Users/Laura/Documents/Uni/Semester_07/Bachelorarbeit/Code/results/evaluationResults.csv')
    x = pd.read_csv('results/evaluationResults.csv')
    
    f = x[(x.Data_Set_Name == 'NWN')]
    
    c = f[(x.Algorithm_Name == algoList[0])]
    c = c.rename(columns={'Value': algoList[0]})
    
    for algo in algoList[1:]:
        h = f[(x.Algorithm_Name == algo)]
        h = h.rename(columns={'Value': algo})
        h = h[[algo]]
        c.reset_index(drop=True, inplace=True)
        h.reset_index(drop=True, inplace=True)
        c = pd.concat([c,h], axis=1)
        
    
    
    ax = c.plot.bar(x=['Measure'], y=algoList)
    
    patches, labels = ax.get_legend_handles_labels()
    
    axes = plt.gca()
    axes.set_ylim([0,1])
    
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

    fig = ax.get_figure()
    fig.tight_layout()
    #fig.savefig('C:/Users/Laura/Documents/Uni/Semester_07/Bachelorarbeit/Code/results/NWN.pdf',bbox_inches='tight')
    fig.savefig('results/NWN.pdf')
