#--------
# Layers
#--------


# import
#--------
import numpy as np
import numpy.random as rd

# Input Layer
#-------------    
    
class Input:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
    # Forward pass
    def forward(self, inputs):
        self.output = inputs
        

# Dense Layer
#------------

class DenseDE:
    def __init__(self, n_neurons, activation, previous, weights=None):
        self.n_neurons = n_neurons
        # reference to previous layer
        self.previous = previous
        self.n_inputs = self.previous.n_neurons
        # Initialize weights and biases (+1)
        self.weights = rd.randn(self.n_inputs + 1, self.n_neurons) if weights is None else weights.copy()
        # initialize activation function
        self.activation = activation

    def forward(self, inputs):
        # keep for derivative
        self.inputs = inputs
        # calc output values from inputs, weights and biases
        z = np.dot(inputs, self.weights[:-1]) + np.reshape(1.0*self.weights[-1], (1,-1))
        # activation
        self.output = self.activation.calc(z)


