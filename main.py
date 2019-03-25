# -*- coding: utf-8 -*-
"""
@author: pijnenburgmgf

Execute the Test Procedures for all pairwise event logs
for the 4 synthetic event logs created by gen_eventlogs_return.py
"""


import numpy as np
import pandas as pd
from scipy import stats # for cumulative distribution function of chi square
import general as gen #general utilities

#To disable some data conversion warnings (by sklearn.preprocessing.StandardScaler):
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)



# read the event logs
elog1 = gen.csv2list('logs/eventlog1.txt')
elog2 = gen.csv2list('logs/eventlog2.txt')
elog3 = gen.csv2list('logs/eventlog3.txt')
elog4 = gen.csv2list('logs/eventlog4.txt')

#####################################################
### Compare the event logs                       ####
#####################################################

def compare_logs(LOGA, LOGB):
    biglog, nfeat, XORjuncs, XORjuncsN = gen.prep_logs(LOGA, LOGB)
    aant_xor = len(XORjuncs)
    ChiSind = [-1]*aant_xor
    degf_adjind = [-1]*aant_xor
    pvalind = [-1]*aant_xor
    prevact = [['V1','V2'],['V1','V2','XOR1', 'V5','V6', 'XOR2']]
    for XORcnt in range(aant_xor):
        ChiSind[XORcnt], degf_adjind[XORcnt], pvalind[XORcnt] = gen.checkXOR(biglog, 
               nfeat, prevact[XORcnt], XORjuncs[XORcnt], XORjuncsN[XORcnt])
        print("Chi Square Statistic: %.8f" % ChiSind[XORcnt])
        print('Degrees of freedom: %d' % degf_adjind[XORcnt])
        print("p-value: %.12f" % pvalind[XORcnt])
    Chi = sum(ChiSind)
    degfr = sum(degf_adjind)
    pvalue =  1 - stats.chi2.cdf(Chi, degfr)
    print('The ultimate p-value is: %f' % pvalue)
    print("++++++++++++++++++++++++++++++++++++++++\n") 


compare_logs(elog1, elog1)   # 1
compare_logs(elog1, elog2)   # 0.000000
compare_logs(elog1, elog3)   # 0.975989
compare_logs(elog1, elog4)   # 0.000000
compare_logs(elog2, elog1)   # 0.000000
compare_logs(elog2, elog2)   # 1
compare_logs(elog2, elog3)   # 0.000000
compare_logs(elog2, elog4)   # 0.057141
compare_logs(elog3, elog1)   # 0.975989
compare_logs(elog3, elog2)   # 0.000000 
compare_logs(elog3, elog3)   # 1
compare_logs(elog3, elog4)   # 0.000000
compare_logs(elog4, elog1)   # 0.000000
compare_logs(elog4, elog2)   # 0.057141 
compare_logs(elog4, elog3)   # 0.000000
compare_logs(elog4, elog4)   # 1

def comp_logs_depend(LOGA, LOGB):
    # in order to merge the logs and cluster the cases on the initial features,
    # we use the functions of the previous test procedure, although they compute
    # some characteristics that we do not need.
    # merge logs
    biglog, biglogtr, nfeat = gen.prep_logs_dep(LOGA, LOGB)
    # clustering
    df = gen.featASdf(biglog, nfeat)
#    numClus = 1    
    numClus = round(len(df)/250)
    
    cluslab = gen.clusterme(DF = df, K = numClus, SEED = 12345)
    tab_list = [None] * numClus
    for clcnt in range(numClus):
        trclust = np.array(biglogtr)[cluslab == clcnt]
        location = [int(x[-1]) for x in np.array(biglog)[cluslab == clcnt]]
        freqtab = gen.get_most_freq_seq(trclust, location)
        freqtab.insert(0, 'Cluster' , clcnt)
        tab_list[clcnt] = freqtab
    tabel = pd.concat(tab_list, ignore_index=True) 
    tabel = tabel.reset_index()
    tabel.rename(columns={'index':'output'}, inplace=True)
    print("++++++++++++++++++++++++++++++++++++++++\n") 
    print("      Cluster Statistics\n") 
    print(tabel)
    p_in = sum(tabel.iloc[:,1:tabel.shape[1]].sum())/len(biglog)    
    print("Percentage of traces that do not belong to the most frequent: %d"
        %(100*(1-p_in)))
    ChiStot, degfree, pvalue = gen.cal_pvaldep(tabel)
    print("Chi Square Statistic: %.8f" % ChiStot)
    print('Degrees of freedom: %d' % degfree)
    print("p-value: %.12f" % pvalue)
    print("++++++++++++++++++++++++++++++++++++++++\n") 
    


comp_logs_depend(elog1, elog1) # 1.0       
comp_logs_depend(elog1, elog2) # 0.000000  
comp_logs_depend(elog1, elog3) # 0.982920 
comp_logs_depend(elog1, elog4) # 0.000000  
comp_logs_depend(elog2, elog1) # 0.000000  
comp_logs_depend(elog2, elog2) # 1.0       
comp_logs_depend(elog2, elog3) # 0.000000  
comp_logs_depend(elog2, elog4) # 0.662541
comp_logs_depend(elog3, elog1) # 0.982920 
comp_logs_depend(elog3, elog2) # 0.000000  
comp_logs_depend(elog3, elog3) # 1.0      
comp_logs_depend(elog3, elog4) # 0.000000 
comp_logs_depend(elog4, elog1) # 0.000000  
comp_logs_depend(elog4, elog2) # 0.662541
comp_logs_depend(elog4, elog3) # 0.000000  
comp_logs_depend(elog4, elog4) # 1.0      
    




