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
            self.biases.append(numpy.random.randn(layer) * numpy.sqrt(2 / last_output))
            last_output = layer


    def softmax(self, a):
        a_exp = numpy.exp(a - numpy.max(a))
        return a_exp / numpy.sum(a_exp, axis=-1, keepdims=True)


    def ReLU(self, a):
        return numpy.where(a >= 0, a, a * 0.05)
    

    def ReLU_derivative(self, a):
        return numpy.where(a >= 0, 1, 0.05)
    

    def forward(self, input_layer):
        input_layer = numpy.atleast_2d(input_layer)

        if input_layer.shape[-1] != self.layers[0]:
            raise ValueError(f"Invalid input for method forward: First layer contains {self.layers[0]} neurons!")
        
        self.outputs = [input_layer]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            self.outputs.append(self.ReLU(self.outputs[-1].dot(w) + b))

        self.outputs.append(self.softmax(self.outputs[-1].dot(self.weights[-1]) + self.biases[-1]))

        return self.outputs[-1]
    

    @classmethod
    def crossover(cls, nn1, nn2):
        child = NeuralNetwork(*nn1.layers)

        for i in range(len(child.weights)):
            mask = numpy.random.randint(0, 2, size=child.weights[i].shape).astype(bool)
            child.weights[i] = numpy.where(mask, nn1.weights[i], nn2.weights[i])

        for i in range(len(child.biases)):
            mask = numpy.random.randint(0, 2, size=child.biases[i].shape).astype(bool)
            child.biases[i] = numpy.where(mask, nn1.biases[i], nn2.biases[i])

        return child
    

    def add_randomness(self, random_amount):
        new_network = NeuralNetwork(*self.layers)

        new_network.weights = [w.copy() for w in self.weights]
        new_network.biases  = [b.copy() for b in self.biases]

        for i in range(len(new_network.weights)):
            noise = numpy.random.randn(*new_network.weights[i].shape) * random_amount
            new_network.weights[i] += noise

        for i in range(len(new_network.biases)):
            noise = numpy.random.randn(*new_network.biases[i].shape) * random_amount
            new_network.biases[i] += noise

        return new_network

        
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
    


if __name__ == "__main__":
    brain = NeuralNetwork(2, 4, 3)
    print(brain.biases)
    brain2 = brain.add_randomness(0.1)
    print(brain2.biases)

