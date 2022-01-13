"""
Preparation of Magic data set
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
import pandas as pd 


# load data
df   = pd.read_csv(r'Experiments\data\magic04.data', header=None)
# checking for missing values
df.isnull().sum() # no missing values

# rename target
df[10] = np.where(df[10]=='g', 1, 0)

# split data to make subset
df1 = df[df[10]==1]
df0 = df[df[10]==0]


ratio1 = df1.shape[0] /df.shape[0]
s_nr1 = int(np.floor(8000*ratio1))

# make subset
df1 = df1.sample(n=s_nr1, random_state=5)
df0 = df0.sample(n=8000-s_nr1, random_state=5)
data = df1.append(df0)

# transfer to numpy array
data = data.values
# divide data into features and target
X, y   = data[:,0:10], data[:,-1].astype('int')
names = ['background', 'signal'] 
target = np.array(names)[y]
#sns.countplot(target)

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

# save data
# s.save(datafin, 'magic')















