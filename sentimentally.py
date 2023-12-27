import pandas as pd 
import numpy as np


class model:
    def __init__(self):
        #setting defaults
        self.epochs = 10
        self.lr = 0.1
        self.layers = []
        self.structure = []
        self.activation = 'sigmoid'

    def addLayer(self, size, activation = 'sigmoid'):
        self.layers.append(Layer(size, activation))
        self.structure.append(size)

class Layer:
    def __init__(self, size, activation = 'sigmoid'):
        self.size = size
        self.activation = activation
        self.weights = np.random.rand(size, 1)
        self.bias = np.random.rand(size, 1)
    
        
    