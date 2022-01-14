
"""
Neural network to see with SGD
"""


import numpy as np
from keras.layers import Input, Dense 
from keras.models import Model
from keras.callbacks import History



class AE_ADAM:
    def __init__(self, n_nodes, activation, optimizer='adam', loss='mse', epoch=5000):
        self.n_nodes    = n_nodes.copy()
        self.activation = activation.copy()
        self.epoch      = epoch
        self.loss       = loss
        self.optimizer  = optimizer
        self.AE         = None
        self.encod      = None
        self.report     = History()
        self.create()
        self.setCompiler()
        
    # model creation
    def create(self):
        # copy nodes
        nodes = self.n_nodes.copy()
        acti  = self.activation.copy()
        # create layer list
        model = []
        # create input layer
        input_lay = Input(shape=(nodes[0],))
        # delete nr of inputs
        del nodes[0]
        # append input layer to set previous
        model.append(input_lay)
        # create all layers according to the n_nodes list with reference to previous layer
        for layer in range(len(nodes)):
            # create new layer
            new = Dense(units=nodes[layer], activation=acti[layer])(model[-1])
            # append new layer to the model
            model.append(new)
        # save the model
        self.AE      = Model(input_lay, model[-1])
        self.encod   = Model(input_lay, model[np.argmin(nodes)+1])

    def getSummary(self):
        self.AE.summary()
        print()
        self.encod.summary()
        
        
    def setCompiler(self):
        try:
            self.AE.compile(optimizer=self.optimizer, loss=self.loss)
        except:
            print('\'sgd\' or \'adam\' and \'mean_sqared_error\'')
            
    def fit(self, X_train, validation=None):
        if validation is None:
            self.AE.fit(X_train, X_train, epochs=self.epoch, callbacks=[self.report])
        else:
            self.AE.fit(X_train, X_train, epochs=self.epoch, validation_data=(validation, validation), callbacks=[self.report])
            
    def predict(self, X_test):
        return self.AE.predict(X_test)
    
    def encoder(self, X_test):
        return self.encod.predict(X_test)
            
    def getHistory(self):
        return self.report.history
    




