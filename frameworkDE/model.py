#-------------------------
# Model for Neural Network
#-------------------------

# imports 
#--------
from frameworkDE.layers import Input, DenseDE

# Model class
#-----------

class ANN:
    # Initialize the model class
    def __init__(self, n_nodes, activationList, weights=None):
        self.n_nodes    = n_nodes.copy()
        self.activation = activationList.copy() 
        self.weights    = weights if weights is None else weights.copy()
        #create model
        self.model      = self.create()
     
    # model creation
    def create(self):
        # create layer list
        model = []
        # create input layer
        self.input_lay = Input(self.n_nodes[0])
        # delete nr of inputs
        del self.n_nodes[0]
        # append input layer to set previous
        model.append(self.input_lay)
        # create all layers according to the n_nodes list with reference to previous layer
        for i_layer in range(len(self.n_nodes)):
           if self.weights is None: # initilize weights
                # create new layer
                new = DenseDE(self.n_nodes[i_layer], self.activation[i_layer], previous=model[-1])
                # append new layer to the model
                model.append(new)
           else: # use given weights
                # create new layer
                new = DenseDE(self.n_nodes[i_layer], self.activation[i_layer], previous=model[-1], weights=self.weights[i_layer])
                # append new layer to the model
                model.append(new)
        # delete input layer -> connected over previous
        del model[0]
        # return the model
        return model

    def forward(self, X, i_start_layer=None, i_stop_layer=None):
        # if there is no start or stop use all layers
        if i_start_layer is None:
            i_start_layer = 0
        if i_stop_layer is None:
            i_stop_layer = len(self.model)
        else:
            i_stop_layer += 1
        # if no layers are in the model
        if i_start_layer >= i_stop_layer:
            print('Add layer to the model!')
            return X
        # feed data into input layer
        self.input_lay.forward(X)
        # go through all layers in the model and perform forward pass
        for layer in self.model[i_start_layer:i_stop_layer]:
            layer.forward(layer.previous.output)
        # now we are in the last layer so return output for loss
        return layer.output
    
    def getWeights(self):
        W = []
        # iterate through layers and attach weights and biases
        for layer in self.model:
            w = layer.weights
            W.append(w)
        # return the list with weights, last row = biases    
        return W



