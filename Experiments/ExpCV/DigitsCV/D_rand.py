
"""
Experiment2: 
Strategy: jDE/rand/1
Data:     handwritten digits data set    
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
# handwritten digits data set
# =============================================================================


# =============================================================================
# Experiment for model Algo comparison
# =============================================================================

#------------------------------------------------------------------------------
# load data --> Be in first folder
#------------------------------------------------------------------------------
df   = s.load('Experiments\Data\optidigits.dat')
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
n_nodes        = [X_scaled[0][0].shape[-1], 16, X_scaled[0][0].shape[-1]]
activationList = [Sigmoid(), Sigmoid()]
loss           = MSE()
niter          = 10000
popSize        = 230
strat          = 'rand' # can change to 'rand' to use basic strategy (k is in DEGL k=radius of neighborhood)
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
# s.save(models, 'Digits_rand')#Experiments/Data/rand
# =============================================================================
xxx = s.load('Digits_rand.dat')







