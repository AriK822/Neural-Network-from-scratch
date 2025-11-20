import numpy



class NeuralNetwork(object):
    def __init__(self, *layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.create_random_layers()


    def create_random_layers(self):
        last_output = self.layers[0]
        for layer in self.layers[1:]:
            self.weights.append(numpy.random.randn(last_output, layer) * numpy.sqrt(2 / last_output))
            self.biases.append(numpy.zeros(layer))
            last_output = layer


    def softmax(self, a):
        a_exp = numpy.exp(a - numpy.max(a))
        return a_exp / numpy.sum(a_exp, axis=-1, keepdims=True)


    def ReLU(self, a):
        return numpy.maximum(0, a)
    

    def ReLU_derivative(self, a):
        return (a > 0).astype(float)
    

    def forward(self, input_layer):
        input_layer = numpy.atleast_2d(input_layer)

        if input_layer.shape[-1] != self.layers[0]:
            raise ValueError(f"Invalid input for method forward: First layer contains {self.layers[0]} neurons!")
        
        self.outputs = [input_layer]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            self.outputs.append(self.ReLU(self.outputs[-1].dot(w) + b))

        self.outputs.append(self.outputs[-1].dot(self.weights[-1]) + self.biases[-1])

        return self.outputs[-1]
    

    def train(self, input_layer, y_true, batch_size = 1, learning_rate = 0.01):
        y_true = numpy.atleast_2d(y_true)
        y_pred = self.forward(input_layer)

        if y_pred.shape != y_true.shape:
            raise ValueError("True values shape does not match input")
        
        self.update_weights = [0 for _ in range(len(self.weights))]
        self.update_biases  = [0 for _ in range(len(self.biases))]
        
        delta = y_pred - y_true
        self.update_weights[-1] = self.outputs[-2].T @ delta
        self.update_biases[-1]  = delta.sum(axis=0, keepdims=True)
        
        for i in range(len(self.weights) - 1)[::-1]:
            delta = (delta @ self.weights[i + 1].T) * self.ReLU_derivative(self.outputs[i + 1])
            self.update_weights[i] = self.outputs[i].T @ delta
            self.update_biases[i] = delta.sum(axis=0, keepdims=True)

        for w, update_w, b, update_b in zip(self.weights, self.update_weights, self.biases, self.update_biases):
            w -= learning_rate * (1 / batch_size) * update_w
            b -= learning_rate * (1 / batch_size) * update_b.flatten()
        

    def save_network(self, name):
        import pickle
        with open(rf'{name} weights.pkl', 'wb') as f:
            pickle.dump(self.weights, f)
        with open(rf'{name} biases.pkl', 'wb') as f:
            pickle.dump(self.biases, f)

        
    def load_network(self, name):
        import pickle
        with open(rf'{name} weights.pkl', 'rb') as f:
            self.weights = pickle.load(f)
        with open(rf'{name} biases.pkl', 'rb') as f:
            self.biases = pickle.load(f)

    
    def __repr__(self):
        return f"( -, {self.layers[0]}), " + ', '.join([str(layer.shape) for layer in self.weights]) + '\n' + f"( -, {self.layers[0]}), " + ', '.join([str(layer.shape) for layer in self.biases])

