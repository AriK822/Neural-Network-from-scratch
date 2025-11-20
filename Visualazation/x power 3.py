import pygame
import sys
import math
import numpy
from random import randint
from time import sleep



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

        # for i in range(len(self.weights)):
        #     print(self.weights[i].shape, self.update_weights[i].shape)
        # print()
        # for i in range(len(self.weights)):
        #     print(self.biases[i].shape, self.update_biases[i].shape)

        for w, update_w, b, update_b in zip(self.weights, self.update_weights, self.biases, self.update_biases):
            w -= learning_rate * (1 / batch_size) * update_w
            b -= learning_rate * (1 / batch_size) * update_b.flatten()

    
    def __repr__(self):
        return f"( -, {self.layers[0]}), " + ', '.join([str(layer.shape) for layer in self.weights]) + '\n' + f"( -, {self.layers[0]}), " + ', '.join([str(layer.shape) for layer in self.biases])
    




x_axis_size = 5
y_axis_size = 5


WIDTH, HEIGHT = 1800, 950
WHITE = (255, 255, 255)
BLACK = (25, 25, 25)
GRAY = (60, 60, 60)
RED_OUTLINE = (100, 70, 70)



def f(x):
    try:
        return nn.forward([x])[0][0]

    except:
        return 0



def draw_axis():
    pygame.draw.line(screen, GRAY, (0, HEIGHT/2), (WIDTH, HEIGHT/2), 2)
    pygame.draw.line(screen, GRAY, (WIDTH/2, 0), (WIDTH/2, HEIGHT), 2)

    if x_axis_size < 50:
        for i in range(x_axis_size + 1):
            pygame.draw.line(screen, GRAY, (i * WIDTH / x_axis_size, HEIGHT / 2 + 5), (i * WIDTH / x_axis_size, HEIGHT / 2 - 5), 1)
        for i in range(y_axis_size + 1):
            pygame.draw.line(screen, GRAY, (WIDTH / 2 - 5, i * HEIGHT / y_axis_size), (WIDTH / 2 + 5, i * HEIGHT / y_axis_size), 1)



def draw_function(func, color = WHITE):
    lx, ly = 0, 0
    for j in range(-x_axis_size * 10, x_axis_size * 10):
        i = j / 10
        cx, cy = WIDTH / 2 + i / x_axis_size * WIDTH, HEIGHT / 2 - (func(i) / y_axis_size * HEIGHT)
        pygame.draw.line(screen, color, (lx, ly), (cx, cy))
        pygame.draw.rect(screen, color, pygame.Rect(cx, cy, 1, 1))
        lx, ly = cx, cy




pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Graph drawer")


nn = NeuralNetwork(1, 20, 20, 10, 1)

for _ in range(50000):
    a = numpy.random.rand(1, 1) * 3 * (numpy.random.randint(0, 2, (1, 1)) * 2 - 1)
    true_a = a ** 3
    nn.train(a, true_a)

    if _ % 1000:
        screen.fill(BLACK)
        draw_axis()
        draw_function(lambda x : x**3, RED_OUTLINE)
        draw_function(f)
        pygame.display.flip()


running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False





pygame.quit()
sys.exit()

