
"""
Experiment1: 
Strategy: jDE/DEGL/SAW
Data:     swiss roll    
"""

# imports
#----------

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from frameworkDE.optimizer import DENN
from frameworkDE.functions import Sigmoid, MSE, calcDims
import Experiments.plot as p
from sklearn.decomposition import PCA 
import pandas as pd
import seaborn as sns
from copy import deepcopy
import Experiments.saveLoad as s


# =============================================================================
# Swiss Roll
# =============================================================================

#------------------------------------------------------------------------------
# load data 
#------------------------------------------------------------------------------
df   = s.load('Experiments\Data\swissroll.dat')
# data X_scaled scaled with min max 
X_scaled, X_split, y_split = df # X_scaled = train, test, vali / y = train, test, vali

#------------------------------------------------------------------------------
# import random seeds for experiment (fixed in all experiments)
#------------------------------------------------------------------------------
seeds = s.load('Experiments\Data\seeds.dat')


# Experiment
#------------------------------------------------------------------------------ 
# 1. [jDE/DEGL/SAW]
#------------------------------------------------------------------------------

# Parameter settings 
#-------------------
n_nodes        = [3, 8, 2, 8, 3]
activationList = [Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid()]
loss           = MSE()
niter          = 10000
popSize        = 100
strat          = 'MERGE' 
jDE            = True
k              = 3
w_scheme       = 'SAW'
pp             = 0.05

# how many dimensions 
calcDims(n_nodes)

# save all trained model in list
models     = []

# experiment run 10 times with 10000 iteration 
for i in range(10):
    # scaled values
    X_tr_mm, X_te_mm, X_va_mm = X_scaled[i]
    # set seed
    rd.seed(seeds[i])
    # create model accoring to Parameters
    AE = DENN(popSize, n_nodes, activationList, loss, niter, strat, jDE=jDE, k=k, w_scheme=w_scheme, p=pp) 
    # do process
    best = AE.evolution(X_tr_mm, X_va_mm, X_te_mm)
    models.append(deepcopy(AE))


#s.save(models, 'Experiments/result_SR/MERGE/SR_MERGE')
# xxx = s.load('Experiments/result_SR/MERGE/SR_MERGE.dat')


