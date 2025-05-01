# Neural-Network-from-scratch
This project contains a NN from scratch in python (just NumPy), with some additional files, demonstrating example usages.

## Features  
- Forward/backpropagation  
- Gradient descent  
- ReLU activation function

## Example
```python
nn = NeuralNetwork(1, 10, 30, 30, 30, 1)  
nn.train(a, true_a, batch_size=64, learning_rate=0.01) 
nn.forward(x)
```

A simple visualization of how the network learns and works on simple functions like sin(x) and x^2 are shown on "Visualazation.py" folder.

Network is trained on the famous Mnist dataset. I got 96% accuracy. You can test the network by opening "Mnist data set/hand written recognizer.py" and using mouse to draw number, pressing "A" to view the prediction, and pressing "Space bar" to reset.

Also I trained it for a simple task like colorizeing images (A real life task). You can view the reaults by running "image colorizer/colorize test.py".

Best wishes, Arman Kiani, CS studen of university of Tabriz, May 1, 2025.
