
"""
Create Swiss Roll data set
"""

# imports
#-----------
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
import numpy as np
import Experiments.preprocessing as pre
import Experiments.saveLoad as s

#=============================================================================
# Swiss roll
#=============================================================================

# create swissroll dataset
#-------------------------
n_samples    = 8000 
noise        = 0.05
random_state = 1401
X, y = datasets.make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)

# split inot train, test and validation
#---------------------------------------
test_ratio  = 0.2

# seeds
see = s.load('Experiments\Data\seeds.dat')
seeds = see[:5]


# 5x2fold-cv
#----------------------- 
dtx = []
dty = []
x_scaled = []
for seed in seeds:
    # apply
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed, shuffle=True)
    # performe cross vlidation
    rskf = KFold(n_splits=2)
    for train_index, vali_index in rskf.split(X_train, y_train):
        X_tr, X_vali = X_train[train_index], X_train[vali_index]
        y_tr, y_vali = y_train[train_index], y_train[vali_index]
        dtx.append((X_tr, X_test, X_vali))
        dty.append((y_tr, y_test, y_vali))
        x_scaled.append(pre.scaler(X_tr, X_test, X_vali))

datafin = [x_scaled, dtx, dty]

# save as dat file
s.save(datafin, "swissroll")










