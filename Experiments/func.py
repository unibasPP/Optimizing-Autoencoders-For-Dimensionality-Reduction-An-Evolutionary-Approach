

# =============================================================================
# #5x2 CV Significance test
# =============================================================================

# imports
import numpy as np
import numpy.random as rd
import pandas as pd
from scipy.stats import t


def test(scores1, scores2):
    # transform to array
    s1, s2 = np.array(scores1), np.array(scores2)
    diff   = s1-s2
    # difference of first fold in first iteration
    p_1_1  = diff[0] 
    # calculate mean for each of the 5 fold = 5 means
    p_bar = [((diff[i] + diff[i+1])/2) for i in range(0, 9, 2)]   
    # compute the variance estimate of each fold = 5 variances
    p_sig = [(np.power((diff[i] - p_bar[t]), 2) + np.power((diff[i+1] - p_bar[t]), 2)) for i, t in zip(range(0, 9, 2), range(5))]
    # calculate the mean of the variances
    sigm = np.mean(np.array(p_sig))
    # compute the t-value as proposed by diettriech
    t_val = p_1_1 / np.sqrt(sigm)
    p_val = 2*(t.cdf(-abs(t_val), 5))
    # create dataframe with all values
    data = p_val#pd.DataFrame([[t_val], [p_val]], columns=[name12])
    # return list with mean1, mean2 
    return data


def pTable(fitTlist):
    # create names
    r_names = ['Rand', 'CTB', 'CTPB', 'DEGL']
    c_names = ['MERGE', 'DEGL', 'CTPB', 'CTB']
    # create dataframe
    data = pd.DataFrame(columns=c_names, index=r_names)
    # create combinations
    comb = [(0, 4), (0, 3), (0, 2), (0, 1), (1, 4), (1, 3), (1, 2), (2, 4), (2, 3), (3, 4)]
    # fill dataframe
    for co in comb:
        x, y = co
        s1, s2 = fitTlist[x], fitTlist[y]
        p_val = test(s1, s2)
        data.iloc[x, np.abs(y-4)] = p_val
    # return table
    return data    
    
        
def summary(fitList):
    # create names
    r_names = ['RAND', 'CTB', 'CTPB', 'DEGL', 'MERGE']    
    c_names = ['Best', 'Median', 'Worst', 'Mean', 'SD']   
    # create dataframe
    data = pd.DataFrame(columns=c_names, index=r_names)    
    for i in range(5):
        fit = fitList[i]
        data.iloc[i, 0] = np.min(fit)
        data.iloc[i, 1] = np.median(fit)
        data.iloc[i, 2] = np.max(fit)
        data.iloc[i, 3] = np.mean(fit)
        data.iloc[i, 4] = np.std(fit)
    # return dataframe
    return data
    
    
    
    


