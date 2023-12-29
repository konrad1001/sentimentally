import numpy as np


class model:
    def __init__(self):
        #setting defaults
        self.epochs = 10
        self.lr = 0.1
        self.layers = []
        self.structure = []
        self.activation = 'sigmoid'

    def addLayer(self, n_inputs, n_outputs, activation='sigmoid'):
        self.layers.append(LayerDense(n_inputs, n_outputs))
        self.layers.append(ActivationLayer(activation))
        self.structure.append([n_inputs, n_outputs])

    def train(self, x, y):
        N = len(x)
        print(N)
        #check if input size matches first layer size
        
        
        if len(x[0]) != self.structure[0][0]:
            raise ValueError("Input size does not match first layer size")

        for epoch in range(self.epochs):
            cost = 0
            for i in range(N):
                #forward pass
                
                output = x[i]
                for layer in self.layers:
                    output = layer.forward(output)
                cost += self.cost(output, y[i])
                error = self.d_cost(output, y[i])
                
                #backward pass
                for layer in reversed(self.layers):
                    error = layer.updateParameters(error, self.lr)
            cost /= N
            print("Epoch: ", epoch, "Cost: ", cost)

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def cost(self, output, y):
        return -np.log(output) if y == 1 else -np.log(1 - output)
    
    def d_cost(self, output, y):
        return np.divide(-1, output) if y == 1 else np.divide(1, 1 - output)
    
    def to_string(self):
        string = ""
        for layer in self.layers:
            string += layer.to_string()
        return string


class Layer:  
    def __init__(self, size):
        self.size = size     
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def d_relu(self, x):
        return 1 if x > 0 else 0
    
    def get_size(self):
        return self.size

    
class LayerDense(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_neurons)
        self.weights = np.random.rand(n_inputs, n_neurons) * 0.01
        self.bias = np.random.rand(1, n_neurons)
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def updateParameters(self, error, lr):
        input_error = np.dot(error, self.weights.T) 
        weights_error = np.dot(self.inputs.T, error)

        self.weights -= lr * weights_error
        self.bias -= lr * error

        return input_error
    
    def to_string(self):
        return "LayerDense: size = " + str(self.size) + ", weights = " + str(self.weights) + ", bias = " + str(self.bias) + "\n"
    
class ActivationLayer(Layer):
    def __init__(self, activation):
        self.activation = activation
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        if self.activation == 'sigmoid':
            return self.sigmoid(inputs)
        elif self.activation == 'relu':
            return self.relu(inputs)
        else:
            return self.sigmoid(inputs)
        
    def updateParameters(self, error, lr):
        return self.d_sigmoid(self.inputs) * error

    def to_string(self):
        return "ActivationLayer: activation = " + self.activation + ", inputs = " + str(self.inputs) + "\n"