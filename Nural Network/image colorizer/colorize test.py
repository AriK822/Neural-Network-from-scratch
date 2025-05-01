from Nns import NeuralNetwork
from PIL import Image
import numpy as np


nn = NeuralNetwork(49, 64, 98, 128, 147)
nn.load_network('colorizer')


image = Image.open(rf"Test cases/test_case6.jpg")

gray_image = image.convert('L')
gray_array = np.array(gray_image)

colorized_image = np.array(image)

p_size = 7

for i in range(gray_image.size[1] // p_size):
    for j in range(gray_image.size[0] // p_size):

        input_neurons = gray_array[i * p_size : i * p_size + p_size, j * p_size : j * p_size + p_size].flatten() / 255
        prediction = nn.forward(input_neurons) * 255
        prediction = np.clip(0, 255, prediction)

        square_rgb = prediction.reshape(p_size, p_size, 3)

        colorized_image[i * p_size : i * p_size + p_size, j * p_size : j * p_size + p_size] = square_rgb


predictied_image = Image.fromarray(colorized_image)
predictied_image.show()

# modified_array = rgb_array

# modified_image = Image.fromarray(modified_array)

# modified_image.save("output.png")

# print("Image processed and saved as output.png")