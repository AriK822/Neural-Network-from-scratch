from Nns import NeuralNetwork
from PIL import Image
import numpy as np


nn = NeuralNetwork(49, 64, 98, 128, 147)

for i in range(50):
    print(f"Proseccing Image: {i}")
    try:
        image = Image.open(rf"Sample_images/t{i}.jpg")
        gray_image = image.convert('L')

        rgb_array = np.array(image)
        gray_array = np.array(gray_image)

        p_size = 7

        for i in range(gray_image.size[1] // p_size):
            for j in range(gray_image.size[0] // p_size):

                train_data = gray_array[i * p_size : i * p_size + p_size, j * p_size : j * p_size + p_size] / 255
                true_data = rgb_array[i * p_size : i * p_size + p_size, j * p_size : j * p_size + p_size] / 255

                if train_data.shape == (p_size, p_size):
                    train_data = train_data.flatten()
                    true_data = true_data.flatten()

                    nn.train(train_data, true_data, 1, 0.0005)
    except:
        pass

    


nn.save_network('colorizer')

# modified_array = rgb_array

# modified_image = Image.fromarray(modified_array)

# modified_image.save("output.png")

# print("Image processed and saved as output.png")