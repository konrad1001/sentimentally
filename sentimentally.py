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
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def activate(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'relu':
            return self.relu(x)
        else:
            return self.sigmoid(x)
    
    
    
class LayerDense(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons)
        self.bias = np.random.rand(1, n_neurons)
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def updateParameters(self, error, lr):
        input_error = np.dot(error, self.weights.T) 
        weights_error = np.dot(self.input.T, error)

        self.weights -= lr * weights_error
        self.bias -= lr * error

        return input_error
    
class ActivationLayer(Layer):
    def __init__(self, activation):
        self.activation = activation
    
    def forward(self, inputs):
        if self.activation == 'sigmoid':
            return self.sigmoid(inputs)
        elif self.activation == 'relu':
            return self.relu(inputs)
        else:
            return self.sigmoid(inputs)
        
    def updateParameters(self, error, lr):
        return self.d_sigmoid(self.inputs) * error
