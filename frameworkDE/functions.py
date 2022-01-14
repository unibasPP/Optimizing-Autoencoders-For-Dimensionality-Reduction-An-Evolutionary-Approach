
#------------------
# Functions for DE
#------------------


# Imports
#--------
import numpy as np
from numba import njit

#--------------------
# Activation Function
#--------------------

# Tanh activation   
class Tanh:
    """
    Hyperbolic tangent
    """    
    @staticmethod
    @njit
    def calc(inputs):
        return np.tanh(inputs)
    
# ReLU activation
class ReLU:
    """
    Rectified linear activation function
    """
    @staticmethod
    @njit
    def calc(inputs):
        # only positive inputs
        return np.maximum(0, inputs)
    
class LeakyReLU:
    @staticmethod
    @njit
    def calc(inputs):
        # only positive inputs
        return np.where(inputs > 0, inputs, inputs*0.01)  
    
# ReLU activation
class ELU:
    """
    Exponential linear unit activation function
    """
    @staticmethod
    @njit    
    def calc(inputs):
        return np.where(inputs>0, inputs, 0.1*(np.exp(inputs)-1))
    
    
class Swish:
    """
    Swish: similar to ReLU
    """
    @staticmethod
    @njit
    def calc(inputs):
        # calc swish return output
        return (inputs*np.power((1 + np.exp(-inputs)), (-1)))
    
    
# Sigmoid activation function
class Sigmoid:
    """
    Sigmoid function for Binary outputs (Binary regression)
    """
    @staticmethod
    @njit
    def calc(inputs):
        # clip inputs to avoid overflow of exp (clip for float32)
        #inputss = np.clip(inputs, -88.72, 88.72) # for float64 +-709.78
        # calc sigmoid return output
        return np.power((1 + np.exp(-inputs)), (-1))
                
# Linear activation
class Linear:
    """
    Linear slope 1
    """    
    @staticmethod
    @njit    
    def calc(inputs):
        return inputs   

#---------------
# Loss Functions
#---------------

class Norm:
    @staticmethod   
    def calc(y_hat, y_true):
        # calc differences
        diff  = (y_true - y_hat)
        return (np.linalg.norm(diff) / diff.shape[0])
        
class MSE:
    @staticmethod
    @njit    
    def calc(y_hat, y_true):
        # calc squared differences
        diff        = np.square(np.subtract(y_true, y_hat))
        return (np.sum(diff) / (diff.shape[0] * diff.shape[1]))
    


#----------------------------
# function to calc dimension
#----------------------------

# dimensions of the problem 
def calcDims(n_nodes):
    D = [((n_nodes[i-1]+1)*n_nodes[i]) for i in range(1, len(n_nodes))]
    return np.sum(D)   



