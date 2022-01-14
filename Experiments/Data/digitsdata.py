"""
Preparation of Optical recognition of handwritten digits data set for 5x2 CV
"""

# imports
#-----------
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import numpy as np
import Experiments.preprocessing as pre
import Experiments.saveLoad as s
import pandas as pd 
import matplotlib.pyplot as plt

# load form sklearn
#-------------------
digits = load_digits()
# check dataset
print(digits.keys())
print(digits.DESCR)
# see shape of the images
print(digits.images.shape)

# Create feature and target arrays
X = digits.data.astype('float32')
y = digits.target


# 5x2cv
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed, shuffle=True, stratify=y)
    # performe cross vlidation
    rskf = KFold(n_splits=2)
    for train_index, vali_index in rskf.split(X_train, y_train):
        X_tr, X_vali = X_train[train_index], X_train[vali_index]
        y_tr, y_vali = y_train[train_index], y_train[vali_index]
        dtx.append((X_tr, X_test, X_vali))
        dty.append((y_tr, y_test, y_vali))
        x_scaled.append(pre.scaler(X_tr, X_test, X_vali))

datafin = [x_scaled, dtx, dty]

# save data --> Experiments/Data/optidigits.dat
# s.save(datafin, 'Experiments/Data/optidigits')















