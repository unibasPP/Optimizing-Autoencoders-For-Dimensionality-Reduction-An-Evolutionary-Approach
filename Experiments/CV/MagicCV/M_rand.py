
"""
Experiment2: 
Strategy: jDE/rand/1
Data:     magic    
"""


# imports
#----------

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from frameworkDE.optimizer import DENN
from frameworkDE.functions import Sigmoid, MSE
import Experiments.plot as p
from sklearn.decomposition import PCA 
import pandas as pd
import seaborn as sns
from copy import deepcopy
import Experiments.saveLoad as s


# =============================================================================
# Magic data set
# =============================================================================

# =============================================================================
# Experiment for model Algo comparison
# =============================================================================

#------------------------------------------------------------------------------
# load data 
#------------------------------------------------------------------------------
df   = s.load('Experiments\Data\magic.dat')
# data X_scaled scaled with min max 
X_scaled, X_split, y_split = df # X_scaled = train, test, vali / y = train, test, vali

#------------------------------------------------------------------------------
# import random seeds for experiment (fixed in all experiments)
#------------------------------------------------------------------------------
seeds = s.load('Experiments\Data\seeds.dat')


# Experiment
#------------------------------------------------------------------------------ 
# 1. [jDE/rand/1]
#------------------------------------------------------------------------------


# Parameter settings 
#-------------------
n_nodes        = [X_scaled[0][0].shape[-1], 8, 3, 8, X_scaled[0][0].shape[-1]]
activationList = [Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid()]
loss           = MSE()
niter          = 10000
popSize        = 230
strat          = 'rand' 
jDE            = True


# save all trained model in list
models     = []

# experiment run 10 times with 10000 iteration 
for i in range(10):
    # scaled values
    X_tr_mm, X_te_mm, X_va_mm = X_scaled[i]
    # set seed
    rd.seed(seeds[i])
    # create model accoring to Parameters
    AE = DENN(popSize, n_nodes, activationList, loss, niter, strat, jDE=jDE) 
    # do process
    best = AE.evolution(X_tr_mm, X_va_mm, X_te_mm)
    models.append(deepcopy(AE))



# =============================================================================
# s.save(models, 'Experiments/result_Magic/rand/Magic_rand')
# =============================================================================
xxx = s.load('Experiments/result_Magic/rand/Magic_rand.dat')





