
"""
Evaluation 
Data: Digits  
"""

# imports
#----------
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import Experiments.plot as p
from sklearn.decomposition import PCA 
import pandas as pd
import Experiments.saveLoad as s
import Experiments.func as cv
from frameworkADAM.AEKeras import AE_ADAM
from frameworkDE.functions import MSE



# =============================================================================
# Digits
# =============================================================================

# load models form Experiements
#------------------------------------
rand  = s.load('Experiments/result_Digits/rand/Digits_rand.dat')
ctb   = s.load('Experiments/result_Digits/CTB/Digits_ctb.dat')
ctpb  = s.load('Experiments/result_Digits/CTPB/Digits_ctpb.dat')
degl  = s.load('Experiments/result_Digits/DEGL/Digits_DEGL.dat')
merge = s.load('Experiments/result_Digits/MERGE/Digits_MERGE.dat')

# all models in a list
Models = [rand, ctb, ctpb, degl, merge]

# =============================================================================
# extract all Test errors values for comparison and save in list
# =============================================================================

# contain all 10 final Test/Train errors for each model
fit_Te = []
fit_Tr = []
# extract fitness values from Testdata
for xxx in Models:
    fitT = [xxx[i].fitTest for i in range(10)]
    fit_Te.append(fitT)
# extract fitness values from training data
for vvv in Models:
    fitO = [vvv[i].fitOpt for i in range(10)]
    fit_Tr.append(fitO)



# calculate all test values according to Diettrich
P_Values  = cv.pTable(fit_Te) 
# calculate summary of training and test errors
SummaryTe = cv.summary(fit_Te)
SummaryTr = cv.summary(fit_Tr)


# =============================================================================
# Convergence plot
# =============================================================================

# extract Trainings report from the 5 models for line plot
# extract index of best value in training
best_tr = [np.argmin(fit) for fit in fit_Tr]
Report = []
for ccc, be in zip(Models, best_tr):
    Report.append(ccc[be].optReport)

# make line plot
p.line(Report, 10000, ylim=17)


# =============================================================================
# Chose best model and compare it to PCA and ADAM
# =============================================================================

# get Data
#----------
df   = s.load('Experiments\Data\optidigits.dat')
# data X_scaled scaled with min max 
X_scaled, X_split, y_split = df # X_scaled = train, test, vali / y = train, test, vali

#------------------------------------------------------------------------------
# import random seeds for experiment (fixed in all experiments)
#------------------------------------------------------------------------------
seeds = s.load('Experiments\Data\seeds.dat')


# =============================================================================
# Reconstruct data with Algorithm
# =============================================================================

# here we chose the model from ctpb
be   = np.argmin(fit_Te[4])
best = merge[be]
# check if the right one 
best.fitTest
# train best fit
best.fitOpt


# make plot of validation and training
#---------------------------------------
p.lin(best.optReport, best.valiReport, ylim=6)


# get corresponding data
#------------------------
X_tr_mm, X_te_mm, X_va_mm = X_scaled[be]
y_tr   , y_te,    y_va    = y_split[be]
# get corresponding seed
#------------------------
seed = seeds[be]

# determine latent space with pca
#--------------------------------
p.PCplot(X_tr_mm)



# test if it is right autoencoder
#--------------------------------
best.evaluate(X_te_mm, idx=1, test=True)

# get latent respresentation
#----------------------------
reducedAE = best.encoder(X_te_mm, test=True)
reconAE   = best.decoder(reducedAE, test=True)  # same as: best.predict(X_te_mm, True)


# Plot data in low dimension
#---------------------------
# chose 20 random pictures
rd.seed(seeds[5])
i_pics = rd.randint(0, 360, 30)
picX = X_te_mm[i_pics]
picy = y_te[i_pics]

# plot originals 
#---------------
p.plotDigits(picX, picy)

# reduced form 
#--------------
#p.plotDred(reducedAE[i_pics], picy)
# Plot of reconstruction
#------------------------
p.plotDigits(reconAE[i_pics], picy)

# =============================================================================
# Estimate a model with ADAM in keras and Reconstruct data
# =============================================================================

# train AE with ADAM
#-------------------
# =============================================================================
# n_nodes   = [64, 16, 64]
# activation = ['sigmoid', 'sigmoid']
# 
# # create model
# AEadam = AE_ADAM(n_nodes, activation, epoch=10000)
# # schow AE and encoder
# AEadam.getSummary()
# # fit the model
# rd.seed(seed)
# AEadam.fit(X_tr_mm, validation=X_va_mm)
# # extract the history
# hist = AEadam.getHistory()
# # save history in a file 
# 
# reducedADAM = AEadam.encoder(X_te_mm)   
# reconADAM   = AEadam.predict(X_te_mm)
# 
# daaa = [reducedADAM, reconADAM]
# 
# 
# #s.save(hist, 'M_ADAM_MERGE')
# #s.save(daaa, 'M_ADAM_MERGE_redrec')
# =============================================================================
hist   = s.load('Experiments/result_Digits/ADAM/M_ADAM_MERGE.dat')
redrec = s.load('Experiments/result_Digits/ADAM/M_ADAM_MERGE_redrec.dat')
# for Reconstruction error
error = MSE()

# get reduced and reconstructed data
reducedADAM, reconADAM = redrec

# reconstruction error with test data
#------------------------------------
error.calc(reconADAM, X_te_mm)
# reconstruction error with train data
#-------------------------------------
hist['loss'][9999]


# line plot to compare convergence of trainig AE with ADAM and CTPB
#-------------------------------------------------------------------
repADAM = hist['loss']
repAE   = best.optReport
# create line plot
#-----------------
p.lineA(repADAM, repAE, 'MERGE', 10000)


# plot training against validation for ADAM
#------------------------------------------
p.lin(hist['loss'], hist['val_loss'], ylim=5)



# plot AE ADAM reconstructed
#---------------------------
# Reconsturcted with ADAM
#---------------------------
p.plotDigits(reconADAM[i_pics], picy)


# =============================================================================
# Reconstruct data with PCA
# =============================================================================

pca = PCA(n_components=16)
pca.fit(X_te_mm)
reducedPCA = pca.transform(X_te_mm)
reconPCA   = pca.inverse_transform(reducedPCA)
error.calc(reconPCA, X_te_mm)

# plot PCA reconstructed
#---------------------------
# Reconsturcted with PCA
#---------------------------
p.plotDigits(reconPCA[i_pics], picy)

















