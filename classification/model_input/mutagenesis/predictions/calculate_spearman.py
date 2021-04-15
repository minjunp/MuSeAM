import sys
import numpy as np
import pandas as pd
from scipy import stats

## Import default table
filename = "../GRCh38_F9_LDLR.2_SORT1.csv"
df = pd.read_csv(filename)
df_F9 = df[df.Element=='F9']
df_LDLR_2 = df[df.Element=='LDLR.2']
df_SORT1 = df[df.Element=='SORT1']


def calculate_spearman(filename, df, type):
    ## Import predictions
    vals = np.loadtxt(filename)

    if type == 'regression':
        labelOne = vals

    if type == 'classification':
        labelZero = vals[:,0]
        labelOne = vals[:,1]

    refs = []
    alts = []
    for count, val in enumerate(labelOne, start = 1):
        if count % 2 == 1:
            refs.append(val)
        else:
            alts.append(val)

    ## Calculate Alternate - Reference
    diff = np.subtract(alts, refs)
    #diff = alts

    ## Calculate Spearman
    trueVals = df.Value.values
    outs = stats.spearmanr(trueVals, diff)
    return outs

"""
F9_outs = calculate_spearman("./MuSeAM_classification_shuffle_199bp/F9_pred.txt", df_F9, 'classification')
LDLR_2_outs = calculate_spearman("./MuSeAM_classification_shuffle_199bp/LDLR_2_pred.txt", df_LDLR_2, 'classification')
SORT1_outs = calculate_spearman("./MuSeAM_classification_shuffle_199bp/SORT1_pred.txt", df_SORT1, 'classification')

print(F9_outs) # SpearmanrResult(correlation=0.08428770829352789, pvalue=0.008160646423861101)
print(LDLR_2_outs) # SpearmanrResult(correlation=0.08775637383974924, pvalue=0.0036894920465111464)
print(SORT1_outs) # SpearmanrResult(correlation=0.09676011717770558, pvalue=2.092656836023756e-05)
"""

#F9_outs = calculate_spearman("./MuSeAM_regression_171bp/F9_pred.txt", df_F9, 'regression') # SpearmanrResult(correlation=0.14463727291956394, pvalue=5.231081311335936e-06)
#LDLR_2_outs = calculate_spearman("./MuSeAM_regression_171bp/LDLR_2_pred.txt", df_LDLR_2, 'regression') # SpearmanrResult(correlation=0.06367044285329343, pvalue=0.035316816050022536)
#SORT1_outs = calculate_spearman("./MuSeAM_regression_171bp/SORT1_pred.txt", df_SORT1, 'regression') # SpearmanrResult(correlation=0.14391911399926927, pvalue=2.200226790681905e-10)

F9_outs = calculate_spearman("./MuSeAM_regression_allData_171bp/F9_pred.txt", df_F9, 'regression') # SpearmanrResult(correlation=0.12652843512965864, pvalue=6.891445908841035e-05)
LDLR_2_outs = calculate_spearman("./MuSeAM_regression_allData_171bp/LDLR_2_pred.txt", df_LDLR_2, 'regression') # SpearmanrResult(correlation=0.12614876742613243, pvalue=2.8832825997966373e-05)
SORT1_outs = calculate_spearman("./MuSeAM_regression_allData_171bp/SORT1_pred.txt", df_SORT1, 'regression') # SpearmanrResult(correlation=0.09811809059366577, pvalue=1.5976972427985172e-05)

print(F9_outs)
print(LDLR_2_outs)
print(SORT1_outs)
