# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 21:24:57 2018

@author: Laura
"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd

#constants for algorithms
COLORBLIND = 'Color-Blind'
ALGO_FAIR = 'FAIR'
ALGO_LFRANKING = 'LFRanking'
ALGO_FELDMAN = 'FeldmanEtAl'
ALGO_FOEIRDPC = 'FOEIR-DPC'
ALGO_FOEIRDTC = 'FOEIR-DTC'
ALGO_FOEIRDIC = 'FOEIR-DIC'
ALGO_LISTNET = 'ListNet'

DIR = 'DIR'
DTR = 'DTR'
RKL = 'rKL'

def plotData():
    
    """
    Plots the data 
    
    Saves a pdf file for each data set with plots for measures NDCG@1, NDCG@5, NDCG@10, MAP, and Fairness@k.
    Since those all have their best value at 1, they can be easily compared
    Additionally, for DIR, DTR, and rKL we save a pdf file for their performance on every data set.
    DIR and DTR show the diviations from 1., rKL has its best value at 0.
    Lastly, we print a pdf with NWN, the overall performance of the different Algorithms across all data set.
    """

    algoList = [COLORBLIND, ALGO_FAIR,ALGO_LFRANKING,ALGO_FELDMAN, ALGO_FOEIRDPC, ALGO_FOEIRDTC, ALGO_FOEIRDIC, ALGO_LISTNET]
    
    x = pd.read_csv('results/evaluationResults.csv')
    
    dataSets = x['Data_Set_Name']

    dataSets = dataSets.drop_duplicates()
    
    extraMeasures = [DIR, DTR, RKL]
    #plots the extra Plots for DIR, DTR, and RKL
    for i in extraMeasures:
        plotExtra(x, algoList, i)
        
    #plots a plot for each data set and NWN
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
        fig.savefig('results/'+value+'.pdf',bbox_inches='tight')
    
def plotExtra(x, algoList, measure):
    
    """
    Plots extra plots for DIR, DTR and rKL
    
    @param x: dataFrame with evaluation results
    @param algoList: List with all algorithms
    @param measure: List with extra measures DIR, DTR and rKL
    
    plot plots for DTR, DIR and rKL 
    """    
    dataFrame = x[(x.Measure == measure)]
    dataFrame = dataFrame[(dataFrame.Data_Set_Name != 'NWN')]
    
    df = dataFrame[(dataFrame.Algorithm_Name == algoList[0])]
    df = df.rename(columns={'Value': algoList[0]})

    df = df.set_index('Data_Set_Name')

    for algo in algoList[1:]:
        h1 = dataFrame[(dataFrame.Algorithm_Name == algo)]
        h1 = h1.rename(columns={'Value': algo})
        h1 = h1[['Data_Set_Name',algo]]
        df = df.join(h1.set_index('Data_Set_Name'))
    
    if measure == DIR or measure == DTR:
        df = pd.concat([df[algoList] - 1],axis=1)
        dirAndDtr = df.plot.bar(y=algoList)
        ymin, ymax = plt.ylim()  # return the current ylim
        plt.ylim(ymin=-1, ymax=1)   # set the ylim to ymin, ymax
        plt.title(measure+' with Fixed Scales')
        dirAndDtr.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        fig_rKL = dirAndDtr.get_figure()
        fig_rKL.tight_layout()
        fig_rKL.savefig('results/'+measure+'.pdf',bbox_inches='tight')
    
    n = df.plot.bar(y=algoList)
    
    if measure == DIR or measure == DTR:
        plt.axhline(0, color='k',linewidth=0.1)
        
    plt.title(measure)
    n.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    fig_rKL = n.get_figure()
    fig_rKL.tight_layout()
    fig_rKL.savefig('results/'+measure+'.pdf',bbox_inches='tight')

