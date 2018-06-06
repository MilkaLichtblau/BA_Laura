# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 21:24:57 2018

@author: Laura
"""


import matplotlib.pyplot as plt
import pandas as pd

#constants for algorithms
COLORBLIND = 'Color-Blind'
ALGO_FAIR = 'FAIR'
ALGO_LFRANKING = 'LFRanking'
ALGO_FELDMAN = 'FeldmanEtAl'
ALGO_FOEIRDPC = 'FOEIR-DPC'
ALGO_FOEIRDTC = 'FOEIR-DTC'
ALGO_FOEIRDIC = 'FOEIR-DIC'

DIR = 'DIR'
DTR = 'DTR'
RKL = 'rKL'

def plotData():

    algoList = [COLORBLIND, ALGO_FAIR,ALGO_LFRANKING,ALGO_FELDMAN, ALGO_FOEIRDPC, ALGO_FOEIRDTC, ALGO_FOEIRDIC]
    
    #x = pd.read_csv('C:/Users/Laura/Documents/Uni/Semester_07/Bachelorarbeit/Code/results/evaluationResults.csv')
    x = pd.read_csv('results/evaluationResults.csv')
    
    dataSets = x['Data_Set_Name']

    dataSets = dataSets.drop_duplicates()
    
    extraMeasures = [DIR, DTR, RKL]
    
    for i in extraMeasures:
        plotExtra(x, algoList, i)
    
    for index, value in dataSets.iteritems():
        
        f = x[(x.Data_Set_Name == value)]
        
        c = f[(f.Algorithm_Name == algoList[0])]
        c = c.rename(columns={'Value': algoList[0]})
        
        for algo in algoList[1:]:
            h = f[(f.Algorithm_Name == algo)]
            h = h.rename(columns={'Value': algo})
            h = h[[algo]]
            c.reset_index(drop=True, inplace=True)
            h.reset_index(drop=True, inplace=True)
            c = pd.concat([c,h], axis=1)
            
        if value != 'NWN':
           
            c = c[(c.Measure != 'rKL')]
            c = c[(c.Measure != 'DIR')]
            c = c[(c.Measure != 'DTR')]
            
            ax = c.plot.bar(x=['Measure'], y=algoList)
            
        else:
        
            ax = c.plot.bar(x=['Measure'], y=algoList)
        
        axes = plt.gca()
        axes.set_ylim([0,1])
        
        plt.title(value)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    
        fig = ax.get_figure()
        fig.tight_layout()
        #fig.savefig('C:/Users/Laura/Documents/Uni/Semester_07/Bachelorarbeit/Code/results/'+value+'.pdf',bbox_inches='tight')
        fig.savefig('results/'+value+'.pdf',bbox_inches='tight')
    
def plotExtra(x, algoList, measure):
    
    rKL = x[(x.Measure == measure)]
    rKL = rKL[(rKL.Data_Set_Name != 'NWN')]
    
    rKL1 = rKL[(rKL.Algorithm_Name == algoList[0])]
    rKL1 = rKL1.rename(columns={'Value': algoList[0]})
    
    for algo in algoList[1:]:
        h1 = rKL[(rKL.Algorithm_Name == algo)]
        h1 = h1.rename(columns={'Value': algo})
        h1 = h1[[algo]]
        rKL1.reset_index(drop=True, inplace=True)
        h1.reset_index(drop=True, inplace=True)
        rKL1 = pd.concat([rKL1,h1], axis=1)
        

    n = rKL1.plot.bar(x=['Data_Set_Name'], y=algoList)
    
    if measure == DIR or measure == DTR:
        plt.axhline(1, color='k')
        
    plt.title(measure)
    n.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    fig_rKL = n.get_figure()
    fig_rKL.tight_layout()
    #fig_rKL.savefig('C:/Users/Laura/Documents/Uni/Semester_07/Bachelorarbeit/Code/results/'+measure+'.pdf',bbox_inches='tight')
    fig_rKL.savefig('results/'+measure+'.pdf',bbox_inches='tight')


#plotData()