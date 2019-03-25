# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:14:50 2018

@author: pijnenburgmgf
This file:
- reads an eventlog in memory as a list of lists
- calculates the Markov Matrix of transition probabilities 
"""

import numpy as np
import pandas as pd
import re # for regular expressions, used in wordcount
from sklearn.feature_selection import f_classif # for feat_sel
from sklearn.preprocessing import StandardScaler # for clustering and feat_sel
from sklearn.mixture import GaussianMixture # for clustering
from scipy import stats # for cumulative distribution function of chi square
from collections import Counter # for determining most frequent traces in get_most_freq_seq
#from matplotlib import pyplot as plt # for visclusters
    


def csv2list(FILE):
     ### read the file
    fh = open(FILE,'r')
    elog = [line.rstrip(' \n') for line in fh]
    fh.close()
    return(elog)

def detNfeats(FIRSTLINE):
    '''
    Takes in a line of the event log and looks how many 
    elements there are before the occurence of the letter 'A'.
    The letter 'A' is special since it signifies the start of the trace.
    FIRSTLINE - a list of strings representing one trace.
    '''
    nfeat = 0
    for i in range(len(FIRSTLINE)):
        s = FIRSTLINE[i]        
        if s != 'A':
            nfeat = nfeat + 1
        else:
            break
    return(nfeat)


def removeFeat(ELOG, NFEAT):
    '''
    Many times we need only the activities of the event log,
    and not the features or location label. This functions throws away all 
    features. 
    ELOG  - a list of strings. Each element is a trace.
    NFEAT - number of initial features
    Output is an event log (list of strings).
    '''    
    # construct the new (reduced) event log    
    elogtr = []
    for tr in range(len(ELOG)):
        actl = ELOG[tr].split()
        outtr = ''
        for act in range(NFEAT, len(actl)): 
            outtr = outtr + ''.join(actl[act])
            # add a space as well
            outtr = outtr + ' '
            if actl[act] == 'Z':
                break
        outtr = outtr[:-1]
        elogtr.append(outtr)   
    return(elogtr)
    
def find_XORjunctions(SEARCHSTRING):
    """
    SEARCHSTRING  - the event log represented as one long string. 
    Output: a string vector containing the XOR-junctions
    """       
    out = list(set(re.findall(r"XOR.", SEARCHSTRING)))
    out.sort()
    return(out)
               
def wordcount(ITEMLIST, SEARCHSTRING):
    '''
    For each element in SEARCHLIST, this function counts
    the occurrences in LONGSTRING.
    ITEMLIST     - list with (string) elements to be searched for
    SEARCHSTRING - the string that has to be searched for occurences of elements
                   of ITEMLIST.
    Output is an integer list the same length and order as ITEMLIST with the
    counts
    '''
    output = [0]*len(ITEMLIST)
    for w in re.findall(r"\w+", SEARCHSTRING):
        if w in ITEMLIST:
            output[ITEMLIST.index(w)] += 1
    return(output)               

def is_float(input):
  try:
    num = float(input)
  except ValueError:
    return False
  return True

def is_int(input):
  try:
    num = int(input)
  except ValueError:
    return False
  return True


def MakeABT(ELOG, NFEAT, PREVACT, XOR, N_XOR):
    '''
    Make an ABT: A pandas dataframe with all lines that contain XOR together 
    with the initial features and a count for all other previous activities.
    For XOR2, we also derive the value of the test after repair.
    The dataframe contains a 'output' column that is the next activity.
    Note that in one trace the same XOR junction may appear multiple times 
    due to loops.
    ELOG        - event log, i.e. a list of traces, a trace being a string
                  of activities separated by blanks
    NFEAT        - number of initial features
    PREVACT     - a list with the names of the previous process activities
    XOR         - name (string) of the XOR-junction
    N_XOR       - the number of times this XOR function appears in the event log
    Output is a pandas data frame.
    '''
    # make the empty dataframe with enough columns and rows:
    featcols = [m+n for m,n in zip(['X']*NFEAT, list(map(str,list(range(NFEAT)))))]
    if XOR == 'XOR1':
        col_names = featcols + PREVACT + ['output'] + ['location_label']
    if XOR == 'XOR2':
        col_names = featcols + ['test_result'] + PREVACT + ['output'] + ['location_label']
    df = pd.DataFrame(0, index = range(N_XOR), columns = col_names)
    
    firstline = ELOG[0].split()
    
    feattype = ['empty']*NFEAT
    for i in range(NFEAT):
        if is_int(firstline[i]) is True:
            feattype[i] = 'int'
        else:
            if is_float(firstline[i]) is True:
                feattype[i] = 'float'
            else:
                feattype[i] = 'string'
    # Loop over the event log and add a row to the df as we encounter XOR:
    cntrdf = 0 # makes sure that we add a new line to df at the right place
    for tr in ELOG:
        # trace to list:
        listtr = tr.split()
        # loop over the trace:
        cntr = 0 # keeps track of what activity we are of this trace
        for act in listtr:
            if act == XOR:
                # add initial features to df
                for i in range(NFEAT):
                    if feattype[i] == 'float':
                        df.iloc[cntrdf,i] = float(listtr[i])
                    if feattype[i] == 'int':
                        df.iloc[cntrdf,i] = int(listtr[i])
                    if feattype[i] == 'string':
                        df.iloc[cntrdf,i] = str(listtr[i]) 
                # if XOR = XOR2 then derive the result of the test outcome
                if XOR == 'XOR2':
                    if listtr[cntr + 1] == 'Z':
                        test_res = 1
                    else:
                        test_res = 0
                    df.iloc[cntrdf, NFEAT] = test_res 
                # calculate counts of previous activities
                prevact_str = ''
                for x in listtr[NFEAT:cntr]:
                    prevact_str = prevact_str + ''.join(x) + ' '
                prevactN = wordcount(PREVACT, prevact_str)
                # add process features to df
                if XOR == 'XOR2':
                    df.iloc[cntrdf,range(NFEAT + 1, NFEAT + 1 + len(PREVACT))] = [int(i) for i in prevactN]
                else:
                    df.iloc[cntrdf,range(NFEAT, NFEAT + len(PREVACT))] = [int(i) for i in prevactN]
                # add output to df, i.e. next activity   
                df.loc[df.index[cntrdf], 'output'] = listtr[cntr+1]
                # add location label to df   
                df.loc[df.index[cntrdf], 'location_label'] = listtr[-1]
                # add df counter since the next encounter of XOR must on new row
                cntrdf += 1
            cntr += 1
    return(df)

def featASdf(BIGLOG, NFEAT):
    ''' 
    Read BIGLOG and save the initial features in a DataFrame
    ELOG        - event log, i.e. a list of traces, a trace being a string
                  of activities separated by blanks
    NFEAT        - number of initial features
    '''
    # make empty data frame
    featcols = [m+n for m,n in zip(['X']*NFEAT, list(map(str,list(range(NFEAT)))))]
    df = pd.DataFrame(0, index = range(len(BIGLOG)), columns = featcols)
    
    # determine type of features
    firstline = BIGLOG[0].split()
    feattype = ['empty']*NFEAT
    for i in range(NFEAT):
        if is_int(firstline[i]) is True:
            feattype[i] = 'int'
        else:
            if is_float(firstline[i]) is True:
                feattype[i] = 'float'
            else:
                feattype[i] = 'string'
    
    # fill dataframe
    for rowcnt in range(len(BIGLOG)):
        tr = BIGLOG[rowcnt]
        listtr = tr.split()
        for featcnt in range(NFEAT):
            if feattype[featcnt] == 'float':
                df.iloc[rowcnt,featcnt] = float(listtr[featcnt])
            if feattype[featcnt] == 'int':
                df.iloc[rowcnt,featcnt] = int(listtr[featcnt])
            if feattype[featcnt] == 'string':
                df.iloc[rowcnt,featcnt] = str(listtr[featcnt]) 
    return(df)
    
def feat_sel(DF):
    '''
    Selects relevant features.
    The function uses univariate feature selection
    DF    - pandas data frame with the last column being the output
    Output is a numpy array containing the names of the relevant features
    '''
    # split dataframe in predictor matrix and output column
    X = DF.iloc[:,range(DF.shape[1]-2)].copy() # all but output and location label
    y = DF.loc[:, 'output'].copy()
    
    # convert string variables to factors for now
    strfeatbool = X.dtypes == 'object'
    featnames = strfeatbool.index.values.tolist() 
    strfeat = [i for indx,i in enumerate(featnames) if strfeatbool[indx]]
    
    for i in strfeat:
        X[i] = X[i].astype('category').cat.codes
        
    # remove constant columns
    X = X.loc[:, (X != X.iloc[0]).any()] 
    
    # standadize to remove columns that are not identical but only differ in scale or
    # an additive constant
    X[list(X)] = StandardScaler().fit_transform(X)
    X = X.round(decimals = 6)
    
    # remove identical columns
    X = X.T.drop_duplicates().T
    
    # select the feature selection function from sklearn
    fscoresorg = f_classif(X,y)[1]
    fscores = np.nan_to_num(fscoresorg)
    boolsel = fscores < 0.2

    # apply this feature selection function on the data
    col_names = np.array(list(X))
    sel_feat = col_names[boolsel]    
    return(sel_feat)
    
    
def clusterme(DF, K, SEED):
    '''
    This function clusters the observations of DF into K clusters.
    DF          - data frame with features that have to be clustered. 
    K           - number of clusters
    SEED        - an integer detemining the seed of the random generator
    
    Output is vector the same length a the number of rows of DF, containing the
    cluster label for each observation.
    Notes on the choice of the clustering algorithm:
    I first tried DBSCAN, but that makes one big cluster if the observations 
    are all 'connected' (i.e. no clear open spaces between clusters) which is
    typically the case for us. Then I tried Spectral clustering, but 
    technically i could not get it working.
    Besides the documentation mentions mainly image examples and states that it 
    is less suitable for a large number of clusters.
    Then I tried Gaussian Mixtures

    '''
    X = str2dummies(DF)
    
    # normalize features    
    dfs = StandardScaler().fit_transform(X)
    # learn    
    gmm=GaussianMixture(n_components=K, covariance_type="full",
                random_state = SEED)
    gmmfit = gmm.fit(dfs)
    labclusters = gmm.predict(dfs)
    return(labclusters)

def str2dummies(DF):
    '''
    Takes all columns of type object (i.e. strings) and category to dummy
    variable columns. A column with k levels is replaced by k - 1 dummies.
    DF - a pandas dataframe
    Output is a pandas dataframe with the categorical / strings variables 
    replaced by dummies.
    '''
    dfout = pd.get_dummies(DF, drop_first=True)
    return(dfout)

def visclusters(DF, LABELS):
    '''
    visualize clusters (for debugging or just nice...)
    Plots the first two coordinates of DF in a scatterplot with the clusters
    a different color.
    DF          - data frame with 2 features that have been used to cluster.
                  These 2 features will be displayed on the axes of the 
                  scatterplot.
    LABELS      - A vector the same length as the rows of DF containing the
                  cluster labels as integers
    ''' 
    X = str2dummies(DF)
    
    labcolors = ['red']*len(X)
    colorlist = ['red','blue','green','yellow','black', 'orange','brown','pink',
            'purple','violet']
    for i in range(len(LABELS)):
        labcolors[i] = colorlist[LABELS[i]]
    plt.scatter(X.iloc[:,0], X.iloc[:,1], color = labcolors)
    plt.show()      
    
def cal_pvalue(OBSFREQ, EXPFREQ, DGFADJ):
    '''
    Calculates the Chi Square statistic and p-value given the dataframes with
    observed and expected counts.
    OBSFREQ -   a pandas dataframe containing the classification variables 
                and the observed counts for two combined event logs.
    EXPFREQ -   Idem to OBSFREQ but now with expected frequencies.
    DGFADJ  -   Degrees of freedom taking into account the removal of some rows
    Output: ChiSquare Statistic and p-value
    '''
    # Calculate the Chi Square statistic
    ChiDF = pd.DataFrame().reindex_like(OBSFREQ)
    elc = OBSFREQ.shape[1]-1 # enerlaatste column
    ChiDF.iloc[:,0:elc] = OBSFREQ.iloc[:,0:elc].copy()
    ChiDF=ChiDF.rename(columns = {'count':'ChiSqLoc'})

    ChiDF['ChiSqLoc']= (OBSFREQ['count']-EXPFREQ['count'])**2/ EXPFREQ['count']
    ChiStot = ChiDF['ChiSqLoc'].sum()
    
    pvalue =  1 - stats.chi2.cdf(ChiStot, DGFADJ)
    return(ChiStot, pvalue)

def calc_expected(OBSFREQ):
    '''
    Calculated the expected frequencies based on the Null Hypotheses of
    independence between output ('output') and location. 
    OBSFREQ -   a pandas dataframe containing the classification variables 
                and the observed counts for a combined event log.
                The classification variables with their literal names are
                + cluster  - 'Cluster'
                + output   - 'output'
                + location - 'location'.
    Output is a pandas dataframe similar to OBSFREQ but now with the 
    expected counts.
    '''
    # calculate subtotals of OBSFREQ:
    rowSums = OBSFREQ.groupby(['Cluster','output']).agg({'count':['sum']})
    subtotsim = OBSFREQ.groupby(['Cluster']).agg({'count':['sum']})
    subtotcol = OBSFREQ.groupby(['Cluster','locatie']).agg({'count':['sum']})
    # nH0 is going to contain the expected number of observation under H0
    nH0 = pd.DataFrame().reindex_like(OBSFREQ)
    elc = OBSFREQ.shape[1]-1 # enerlaatste column
    nH0.iloc[:,0:elc] = OBSFREQ.iloc[:,0:elc].copy()
    # fill nH0
    for i in range(OBSFREQ.shape[0]):
        sc = OBSFREQ.loc[i,'Cluster']
        ta = OBSFREQ.loc[i,'output']
        lo = OBSFREQ.loc[i,'locatie']
        rs = rowSums.query('Cluster == @sc and output == @ta').iloc[0,0]
        cs = subtotcol.query('Cluster == @sc and locatie == @lo').iloc[0,0]
        ts = subtotsim.query('Cluster == @sc').iloc[0,0]
        nH0.loc[i,'count'] = rs * cs / ts
    return(nH0)

def prune_obs_freq(FREQ_ORIG):
    '''
    The Chi square is only valid if the expected frequencies of all cells are 1 
    or larger and 80% of the cells have an expected cell count of 5 or larger. 
    For this reason we remove all rows which have a rowsum less than
    4 times the number of columns.
    FREQ_ORIG - a pandas dataframe containing the classification variables 
                and the observed counts for two combined event logs.
                The classification variables with their literal names are
                + cluster  - 'Cluster'
                + output   - 'output'
                + location - 'location'.
    Output is a similar data frame as the input, but now with possible
    some rows (i.e. combination of Cluster and output) deleted.
    '''
    rowMeans = FREQ_ORIG.groupby(['Cluster','output']).agg({'count':['mean']})
    rowMeansdf = rowMeans.stack([0]).reset_index()
    keeprows = rowMeansdf.loc[rowMeansdf['mean'] >= 4,['Cluster', 'output']]
    ctf_adj = pd.merge(FREQ_ORIG, keeprows, on = ['Cluster','output']) 
    nrow_del = len(rowMeans) - len(keeprows)
    return(ctf_adj, nrow_del)
    
def get_freqtab(DF):
    '''
    Tabulates and computes counts of the data set that belongs to one 
    XOR-junction.
    DF - pandas data frame containing three columns:
         * output, i.e. output of XOR-junction (string)
         * Cluster, i.e. cluster number (integer)
         * locatie (integer)
    '''
    conttab = pd.crosstab([DF['Cluster'],DF['output']], DF['locatie'], dropna = False)
    # conttabflaf is the flattened version of conttab: easier for calculations
    conttabflat = conttab.stack([0]).reset_index()
    conttabflat.columns = ['Cluster','output','locatie', 'count']
    return(conttab, conttabflat)
   
def prep_logs(LOGA, LOGB):
    '''
    General preparations of the event logs. First prep_logs_dep is called.
    Additionally the XOR junctions are found and the numbers they occur are 
    recorded.
    LOGA -  lists of strings. Each string contains the features of a case and 
            the trace.
    LOGB -  lists of strings. Each string contains the features of a case and 
            the trace.
    Output are the total log (biglog), the number of features (nfeat), the 
    XOR-junctions (XORjuncs), and the number of times each XOR-junction is 
    passed in both event logs.
    '''
    biglog, biglogtr, nfeat = prep_logs_dep(LOGA, LOGB)
    
    ### convert event log to one large string for efficient search.
    eltr_str = '' 
    for x in biglogtr:
        eltr_str = eltr_str + ''.join(x) + ' '

    XORjuncs = find_XORjunctions(eltr_str)
    XORjuncsN = wordcount(list(XORjuncs), eltr_str) 
    return(biglog, nfeat, XORjuncs, XORjuncsN)
    
def prep_logs_dep(LOGA, LOGB):
    '''
    Preparations on the event logs. Among others the logs are merged
    and the traces are split from the features.
    LOGA -  lists of strings. Each string contains the features of a case and 
            the trace.
    LOGB -  lists of strings. Each string contains the features of a case and 
            the trace.
    Output is the merged log, the log without features and the number of 
    features.
    '''
    loga2 = LOGA.copy()
    for i in range(len(LOGA)):
      loga2[i] = LOGA[i] + ' 0'

    logb2 = LOGB.copy()
    for i in range(len(LOGB)):
      logb2[i] = LOGB[i] + ' 1'
    
    biglog = loga2 + logb2 
    biglog.sort()
        
    # Strip the features and location label from the event logs 
    nfeat = detNfeats(biglog[0].split())
    biglogtr = removeFeat(biglog, nfeat)
    return(biglog, biglogtr, nfeat)

def checkXOR(BIGLOG, NFEAT, PREVACT, XOR, NXOR):
    '''
    This is the main program for one XOR-junction. It calls all other functions
    that together do the following steps:
        * make ABT for this XOR-junction
        * feature selection
        * clustering
        * crosstab
        * checking assumptions
        * calculate Chi Square statistics
    BIGLOG      - lists of strings. Each string contains the features of a case
                and the trace.
    NFEAT       - number of features of a case
    PREVACT     - list of strings containing the previous activities in the 
                  process model of the XOR-junction (without start activity).
                  This to avoid taking into account future information. A better
                  way is to include time stamps in the event logs and use all
                  information that is available before the time stamp of the 
                  XOR-junction. But this more elborate approach is not needed 
                  for this simple example
    XOR         - name of the XOR-junction 
    NXOR        - number of times this XOR-junction is passed in BIGLOG.            
    Output is the Chi Square Statistic, the degrees of freedom as well as the
    p-value.
    '''    
    # make ABT
    df = MakeABT(BIGLOG, NFEAT, PREVACT, XOR, NXOR)
    # find relevant features
    sel_feat = feat_sel(df)
    # cluster the observations to define 'similar cases'
    if sel_feat.size != 0:
        maxsize = len(df.groupby(list(sel_feat), as_index=False).size())
        numClus = min(round((NXOR)/100), maxsize)
        labclusters = clusterme(DF = df[sel_feat], K = numClus, SEED = 12345)
        df2 = pd.DataFrame({'Cluster': labclusters, 'output': df['output'], 
                            'locatie': df.loc[:,'location_label']})         
    else:
        df2 = pd.DataFrame({'Cluster': pd.Series([0]*len(df)),
                              'output': df['output'], 
                              'locatie': df.loc[:,'location_label']})     
    # crosstab
    conttab, conttabflat = get_freqtab(df2)
    # pruning if an assumption of the Chi Square test is violated    
    ctf_adj, nrow_del = prune_obs_freq(conttabflat)
    # compute expected counts
    nH0 = calc_expected(ctf_adj)
    # Check another assumption of Chi square test: 80% of expected number >= 5 
    check_assumption = np.mean(nH0['count'] >= 5)*100
    print("Percentage of rows with expected counts of 5 or higher: %d" 
          %check_assumption) # if > 0.8 is okay  
    # calculate p-value
    ntar = len(pd.unique(df2['output']))
    nloc = len(pd.unique(df2['locatie']))
    degfree_adj = numClus*(ntar - 1)*(nloc - 1) - nrow_del * (nloc - 1)
    ChiStot, pvalue = cal_pvalue(ctf_adj, nH0, degfree_adj)
    return(ChiStot, degfree_adj, pvalue)

def get_most_freq_seq(CLUS, LOC):
    '''
    Returns all traces that occur more than x times in CLUS
    CLUS    - numpy array where each element is a trace (i.e. string)
    LOC     - integer list of the same length as CLUS. Each element represents 
              the location of the corresponding element of CLUS
    Output is a pandas dataframe. The rows are the frequent traces (column 
    'output') and in other columns are the frequencies for all locations
    '''
    nloc = len(set(LOC))

    # determine what traces are the most frequent, using data of all locations
    words_to_count = (word for word in CLUS if word[:1].isupper())
    seqfreq = Counter(words_to_count).most_common()
    
    vals = [x[0] for x in seqfreq]
    freqs = np.array([x[1] for x in seqfreq])
    freqseq = np.array(vals)[freqs > 20]
    
    out = pd.DataFrame(index = freqseq)
    for loccnt in range(nloc):
        clusloc = CLUS[np.array(LOC) == loccnt]
        # compute frequencies for this location
        w2c_loc = (word for word in clusloc if word[:1].isupper())
        freqloc = Counter(w2c_loc)
        df_fl = pd.DataFrame(index = list(freqloc))
        df_fl['loc' + str(loccnt)] = list(freqloc.values())
        # select only the most frequent ones as computed from the whole log:
        tab = df_fl.loc[df_fl.index.isin(freqseq)]
        out = tab.join(out) 
    return(out)
    
def cal_pvaldep(TABEL):
    '''
    TABEL - a dataframe representing a contingency table as described in 
            Table 2 of the paper.
    This function first puts the contingency table in a format better
    suited for computation. Then it calls 'cal_pvalue' to get the p-value.
    Output in the value of the Chi square statistic, the degrees of freedom
    and the p value.
    '''
    # reformat tabel
    nloc = TABEL.shape[1] - 2
    tab_locl = [None] * nloc
    for loccnt in range(nloc):
        tabloc = TABEL.iloc[:,[0,1,2 + loccnt]]
        tabloc = tabloc.rename(columns={ tabloc.columns[2]: "count" })
        tabloc.insert(2,'locatie', loccnt)
        tab_locl[loccnt] = tabloc 
    tabfl = pd.concat(tab_locl, sort = False, ignore_index=True)
    # compute expected counts
    nH0 = calc_expected(tabfl)
    print(nH0)
    check_assumption = np.mean(nH0['count'] >= 5)*100
    print("Percentage of rows with expected counts of 5 or higher: %d" 
          %check_assumption) # if > 0.8 is okay  
    # calculate p-value
    degfree = (len(TABEL) - len(pd.unique(TABEL['Cluster'])))*(nloc - 1)
    ChiStot, pvalue = cal_pvalue(tabfl, nH0, degfree)
    return(ChiStot, degfree, pvalue)
    
    