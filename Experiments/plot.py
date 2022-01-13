

"""

Plots

"""


# imports 
#---------
from sklearn.decomposition import PCA 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits import mplot3d
import numpy as np


# plot functions
#---------------
# plot data
#-----------
# plot optical recognition of handwritten digits
def plotDigits(X, labels):
    n_row = int(np.ceil(X.shape[0]/5))
    n_col = int(5)
    fig, axes = plt.subplots(n_row, n_col, figsize=(1.5*n_col,2*n_row))
    for i in range(len(X)):
        ax = axes[i//n_col, i%n_col]
        ax.imshow(X[i].reshape(8, 8), cmap='gray_r')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()


# plot reduce digits    
def plotDred(X, labels):
    n_row = int(np.ceil(X.shape[0]/5))
    n_col = int(5)
    fig, axes = plt.subplots(n_row, n_col, figsize=(1.5*n_col,2*n_row))
    for i in range(len(X)):
        ax = axes[i//n_col, i%n_col]
        ax.imshow(X[i].reshape(4, 4), cmap='gray_r')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()

# general 2D Plot
def plot2D(X_train, y_train, names,n_classes, title, vtitle):
  plt.figure()
  colors = ['navy', 'turquoise', 'darkorange', 'darkgreen', 'darkmagenta', 'red']
  lw = 2
  for color, i, target_name in zip(colors, range(n_classes), names):
      plt.scatter(X_train[(y_train == i), 0], X_train[(y_train == i), 1], color=color, alpha=0.5, lw=lw,
                  label=target_name)
  plt.legend(loc='best', shadow=False, scatterpoints=1)
  plt.title(title)  
  plt.xlabel(vtitle + "1")
  plt.ylabel(vtitle + "2")
  plt.show()
  
  
# scatter plot 3d
def plot3D(reduced, y_train,target_names, n_class, title, vtitle):
    fig = plt.figure(figsize=(4,4))
    ax  = fig.add_subplot(111, projection='3d')
    colors = ['navy', 'turquoise', 'darkorange', 'darkgreen', 'darkmagenta', 'red']
    for l, i, c in zip(target_names, range(n_class), colors):
        ax.scatter(xs = reduced[(y_train==i),0], ys = reduced[(y_train==i),1], zs = reduced[(y_train==i),2], c=c, label=l)
    ax.set_title(title)
    ax.set_xlabel(vtitle+'1')
    ax.set_ylabel(vtitle+'2')
    ax.set_zlabel(vtitle+'3')
    ax.legend(loc='best')
    plt.show()

# plot SwissRoll
def plotSR(X, colors, subtitle):
    fig = plt.figure(figsize=(4,4))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(xs = X[:,0], ys = X[:,1], zs = X[:,2], c=colors, cmap=plt.cm.jet)
    ax.set_title('Swiss roll'+'\n'+subtitle)
    plt.show()
    
def plotSR2D(reduced, colors, method, subtitle):
    plt.scatter(reduced[:,0], reduced[:,1], c=colors, cmap=plt.cm.jet)
    plt.title('Swiss roll: ' + method)
    plt.xlabel(subtitle+'1')
    plt.ylabel(subtitle+'2')
    plt.show()
    


# ploting the % of variance explained by each dimension
def PCplot(data):
    pca     = PCA()
    pcadat  = pca.fit_transform(data) 
    var     = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var = np.cumsum(var)
    plt.plot(cum_var, linewidth=2)
    plt.grid(color='lightgrey')
    plt.ylabel('Cumultative explained variance')
    plt.xlabel('n_components')  
    plt.title('Percentage of variance explained by each feature')
    plt.show()
    
# line for all mutationstrategies    
def line(optReportList, iteration=10000, ylim=None):
    # create variable for iterations
    x = np.array([i for i in range(iteration)])
    # create y for lines
    labels = ["Rand", "CTB", "CTPB", "DEGL", "MERGE"]
    # create line plot
    fig,ax=plt.subplots()
    for report, i in zip(optReportList, range(5)):
        ax.plot(x, np.array(report)*100, label=labels[i])
    ax.legend()
    if ylim is not None:
        ax.set_ylim(0, ylim)    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title('Convergence of the training errors')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Reconstruction error')
    plt.show()

# line plot with ADAM
def lineA(optADAM, optAE, nameAE, iteration=10000, ylim=None):
    # create variable for iterations
    x = np.array([i for i in range(iteration)])
    # create y for lines
    labels = ["ADAM", nameAE]
    # create line plot
    fig,ax=plt.subplots()
    ax.plot(x, np.array(optADAM)*100, label=labels[0])
    ax.plot(x, np.array(optAE)*100, label=labels[1])
    ax.legend()
    if ylim is not None:
        ax.set_ylim(0, ylim)    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title('Convergence of the training errors')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Reconstruction error')
    plt.show()
    
def lineAllA(optReportList, iteration, ylim=None):
    # create variable for iterations
    x = np.array([i for i in range(iteration)])
    # create y for lines
    labels = ["Rand", "CTB", "CTPB", "DEGL", "MERGE", "ADAM"]
    # create line plot
    fig,ax=plt.subplots()
    for report, i in zip(optReportList, range(6)):
        ax.plot(x, np.array(report)*100, label=labels[i])
    ax.legend()
    if ylim is not None:
        ax.set_ylim(0, ylim)    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title('Convergence of the training errors')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Reconstruction error')
    plt.show()
    
def lin(optReport, valiReport, iteration=10000, ylim=None):
    # create variable for iterations
    x = np.array([i for i in range(iteration)])
    # create y for lines
    labels = ["Training", "Validation"]
    # create line plot
    fig,ax=plt.subplots()
    ax.plot(x, np.array(optReport)*100, label=labels[0])
    ax.plot(x, np.array(valiReport)*100, label=labels[1])
    ax.legend()
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title('Training and validation errors')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Reconstruction error')
    plt.show()


# show correlation
#-----------------
def corrPlot(data):
    # create columnnames for panda dataframe
    colnames = ['col%d'%i for i in range(len(data[1]))]
    # create dataframe
    df = pd.DataFrame(data, columns=colnames)
    # calculate the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle; True = do NOT show
    mask = np.zeros_like(corr, dtype='bool')
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # plot the heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmax=0.3, vmin=-0.3, center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
    plt.show()
    

def corrcoef(data):
    corr = np.corrcoef(data.T)
    coef = []
    for i in range(1, corr.shape[0]):
       coef += list(corr[i,:i])
    # return coefs
    return np.sum(np.square(coef))


