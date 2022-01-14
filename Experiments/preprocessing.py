# -*- coding: utf-8 -*-
"""
Preprocessing
"""

# imports
#---------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# train test validation split
#----------------------------
def t_t_v(X, y, tr_r=0.7, tes_r=0.15, val_r=0.15, rand_state=42):
    # split inot train, test and validation
    #---------------------------------------
    train_ratio = tr_r
    test_ratio  = tes_r
    vali_ratio  = val_r
    # apply
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=rand_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=(vali_ratio/(train_ratio+test_ratio)), random_state=rand_state)
    # return 
    return (X_train, X_test, X_valid, y_train, y_test, y_valid)

# minmaxscaler
#--------------
def scaler(X_train, X_test, X_valid, MinMax=True):
    if MinMax:
        mm       = MinMaxScaler()
        # fit only on trian data
        mm.fit(X_train)
        # applying transformation to the data
        X_train_mm = mm.transform(X_train)
        X_test_mm  = mm.transform(X_test) 
        X_vali_mm  = mm.transform(X_valid)   
        # return 
        return (X_train_mm, X_test_mm, X_vali_mm)
    else:
        mm       = StandardScaler()
        # fit only on trian data
        mm.fit(X_train)
        # applying transformation to the data
        X_train_mm = mm.transform(X_train)
        X_test_mm  = mm.transform(X_test) 
        X_vali_mm  = mm.transform(X_valid)
        return (X_train_mm, X_test_mm, X_vali_mm)

