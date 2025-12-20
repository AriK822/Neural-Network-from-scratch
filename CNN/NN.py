import numpy as np
from numpy.typing import NDArray
from typing import Callable, Any, Optional
import pickle



class ActivationFuncs:
    @classmethod
    def no_activation(cls, array:NDArray) -> NDArray:
        return array


    @classmethod
    def ReLU(cls, array:NDArray) -> NDArray:
        return np.maximum(0, array)
    

    @classmethod
    def ReLU_derivative(cls, array:NDArray) -> NDArray:
        return (array > 0).astype(float)
    
    
    @classmethod
    def softmax(cls, array:NDArray) -> NDArray:
        a_exp = np.exp(array - np.max(array))
        return a_exp / np.sum(a_exp, axis=-1, keepdims=True)
    

    @classmethod
    def derivative_func(cls, func:Callable) -> Callable:
        if func == ActivationFuncs.no_activation or func == ActivationFuncs.softmax:
            return ActivationFuncs.no_activation
        
        if func == ActivationFuncs.ReLU:
            return ActivationFuncs.ReLU_derivative 

        raise TypeError("Activation func not supported!")



class InputLayer:
    def __init__(self, neurons:int):
        self.neurons = neurons



class NNLayer:
    def __init__(self, neurons:int, activation_function = ActivationFuncs.ReLU):
        self.neurons = neurons
        self.activation_function = activation_function


    def init_values(self, last_layer_neurons:int):
        self.weights = np.random.randn(last_layer_neurons, self.neurons) * np.sqrt(2 / last_layer_neurons)
        self.biases = np.zeros(self.neurons)


    def forward(self, input_array:NDArray) -> NDArray:
        self.input_array = input_array
        return self.activation_function(input_array.dot(self.weights) + self.biases)
    

    def backwards(self, delta:NDArray, batch_size = 1, learning_rate = 0.01) -> NDArray:
        activation_derivative = ActivationFuncs.derivative_func(self.activation_function)
        down_stream = (delta @ self.weights.T) * activation_derivative(self.input_array)
        self.weights -= learning_rate * (1 / batch_size) * self.input_array.T @ delta
        self.biases -= learning_rate * (1 / batch_size) * delta.sum(axis=0, keepdims=True).flatten()
        return down_stream



class NeuralNetwork:
    def __init__(self, *layers:InputLayer|NNLayer):
        self.layers = layers
        if not isinstance(self.layers[0], InputLayer): raise TypeError("NN must start with layer of type: InputLayer!")
        last_output = self.layers[0].neurons
        for layer in self.layers:
            if isinstance(layer, NNLayer):
                layer.init_values(last_output)
                last_output = layer.neurons
    

    def forward(self, input_layer:NDArray|list) -> NDArray:
        input_layer = np.atleast_2d(input_layer)
        if input_layer.shape[-1] != self.layers[0].neurons:
            raise ValueError(f"First layer contains {self.layers[0].neurons} neurons!")
        
        for layer in self.layers:
            if isinstance(layer, NNLayer):
                input_layer = layer.forward(input_layer)

        return input_layer
    

    def backwards(self, input_layer:NDArray|list, y_true:NDArray|list, 
                  batch_size = 1, learning_rate = 0.01) -> NDArray:
        y_true = np.atleast_2d(y_true)
        y_pred = self.forward(input_layer)

        if y_pred.shape != y_true.shape:
            raise ValueError("True values shape does not match input")
        
        delta = y_pred - y_true
        for layer in self.layers[::-1]:
            if isinstance(layer, NNLayer):
                delta = layer.backwards(delta, batch_size, learning_rate)

        return delta
    

    def save_values(self, name = "values.nn"):
        with open(name, "wb") as f: pickle.dump(self.layers, f)

    
    def load_values(self, name = "values.nn"):
        with open(name, "rb") as f: self.layers = pickle.load(f)

    

if __name__ == "__main__":
    nn = NeuralNetwork(
        InputLayer(3),
        NNLayer(10),
        NNLayer(10),
        NNLayer(3, activation_function=ActivationFuncs.no_activation),
    )

    for _ in range(10000):
            nn.backwards([[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
                         batch_size=3, learning_rate=1e-1)

    print(nn.forward([1, 2, 3]))
