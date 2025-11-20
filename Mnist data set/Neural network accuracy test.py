import numpy
import pandas
from neural_network import NeuralNetwork

nn = NeuralNetwork(28 * 28, 128, 64, 32, 10)
nn.load_network('mnist')


df = pandas.read_csv('mnist_test.csv')
data = df.to_numpy()

labels = data[:, 0]
images = data[:, 1:]

def num_to_prediction(n):
    a = numpy.zeros((len(n), 10), dtype=int)
    for i, n in enumerate(n):
        a[i][n] = 1
    return a



def save_image(pxl_list, yt, yp, i):
    from PIL import Image

    width = 28
    height = 28

    image_data = numpy.array(pxl_list).reshape((height, width))
    image_data = (image_data * 255).astype(numpy.uint8)

    img = Image.fromarray(image_data, mode='L')

    path = f"wrong_predictioons/T({yt}) P({yp})      n{i}.png"
    img.save(path)





test_size = 9999
got_right = 0

for i in range(test_size):
    true_value = labels[i]
    input_neurons = images[i]

    prediction = nn.forward(input_neurons).argmax()

    if prediction == true_value:
        got_right += 1
    else:
        save_image(input_neurons, true_value, prediction, i)


print(f"Accuracy: {round(got_right / test_size * 100, 3)}")

