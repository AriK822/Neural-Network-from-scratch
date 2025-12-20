import numpy as np
from typing import Optional, Callable
from numpy.typing import NDArray
from enum import Enum, auto
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



class PoolType(Enum):
    maxpooling = auto()
    minpooling = auto()
    averagepooling = auto()



class PoolLayer:
    def __init__(self, type = PoolType.maxpooling, filter_shape = (2, 2),
                 padding = 0, stride = 2):
        self.type = type
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride


    def forward(self, input_array:NDArray) -> NDArray:
        self.input_array = input_array
        batch_size, ih, iw, channels = input_array.shape
        fh, fw = self.filter_shape

        oh = (ih + self.padding * 2 - fh) // self.stride + 1
        ow = (iw + self.padding * 2 - fw) // self.stride + 1

        output = np.zeros((batch_size, oh, ow, channels), dtype=np.float32)

        for i, image in enumerate(input_array):
            for oy in range(oh):
                for ox in range(ow):
                    x, y = ox * self.stride - self.padding, oy * self.stride - self.padding
                    patch = image[max(0, y):min(ih, y+fh), max(0, x):min(iw, x+fw)]

                    if self.type == PoolType.maxpooling:
                        output[i, oy, ox] = np.max(patch, axis=(0, 1))
                    elif self.type == PoolType.minpooling:
                        output[i, oy, ox] = np.min(patch, axis=(0, 1))
                    elif self.type == PoolType.averagepooling:
                        output[i, oy, ox] = np.mean(patch, axis=(0, 1))
                    else:
                        raise TypeError("Pool type not supported!")
        
        return output
    

    def backward(self, delta:NDArray) -> NDArray:...



class InputLayer:
    def __init__(self, *shape:int):
        self.shape = shape



class ConvolutionalLayer:
    def __init__(self, filter_count:int, filter_shape:tuple[int, ...] = (3, 3), 
                 padding:int = 1, stride:int = 1,
                 activation_function = ActivationFuncs.ReLU):
        self.filter_shape = filter_shape
        self.filter_count = filter_count
        self.padding = padding
        self.stride = stride
        self.activation_function = activation_function

    
    def init_values(self, last_layer_channel:int):
        fan_in = self.filter_shape[0] * self.filter_shape[1] * last_layer_channel
        self.filter_shape = (*self.filter_shape, last_layer_channel)
        self.filters = np.random.randn(self.filter_count, *self.filter_shape) * np.sqrt(2/fan_in)
        self.biases = np.zeros(self.filter_count)


    def forward(self, input_array:NDArray) -> NDArray:
        self.input_array = input_array
        batch_size, ih, iw, channels = input_array.shape
        fh, fw, _ = self.filter_shape

        oh = (ih + self.padding * 2 - fh) // self.stride + 1
        ow = (iw + self.padding * 2 - fw) // self.stride + 1

        output = np.zeros((batch_size, oh, ow, self.filter_count), dtype=np.float32)

        for i, image in enumerate(input_array):
            for oy in range(oh):
                for ox in range(ow):
                    x, y = ox * self.stride - self.padding, oy * self.stride - self.padding
                    patch = image[max(0, y):min(ih, y+fh), 
                                  max(0, x):min(iw, x+fw)]
                    f = self.filters[:, max(0,-y):max(0,-y)+patch.shape[0], 
                                        max(0,-x):max(0,-x)+patch.shape[1]]
                    output[i, oy, ox] = np.sum(patch * f, axis=(1, 2, 3)) + self.biases
        
        self.z = output.copy()
        return self.activation_function(output)
    

    def backward(self, delta:NDArray, learning_rate = 1e-4) -> NDArray:
        batch_size, ih, iw, channels = self.input_array.shape
        fh, fw, _ = self.filter_shape
        _, delta_h, delta_w, _ = delta.shape

        biases_update = np.sum(delta, axis=(0, 1, 2))
        weights_update = np.zeros_like(self.filters)
        down_stram = np.zeros_like(self.input_array)

        activation_derivative = ActivationFuncs.derivative_func(self.activation_function)
        delta = delta * activation_derivative(self.z)

        for i, image in enumerate(self.input_array):
            for y in range(delta_h):
                for x in range(delta_w):
                    sx, sy = x * self.stride - self.padding, y * self.stride - self.padding

                    for f in range(self.filter_count):
                        patch = image[max(0, sy):min(ih, sy+fh),
                                      max(0, sx):min(iw, sx+fw)]
                        weights_update[f, max(0,-sy):max(0,-sy)+patch.shape[0],
                                          max(0,-sx):max(0,-sx)+patch.shape[1]] += \
                        delta[i, y, x, f] * patch

                        down_stram[i, max(0, sy):min(ih, sy+fh), 
                                      max(0, sx):min(iw, sx+fw)] += \
                        delta[i, y, x, f] * \
                        self.filters[f, max(0,-sy):max(0,-sy)+patch.shape[0], 
                                        max(0,-sx):max(0,-sx)+patch.shape[1]]
                    
        self.biases -= (learning_rate / batch_size) * biases_update
        self.filters -= (learning_rate / batch_size) * weights_update
        return down_stram


class CNN:
    def __init__(self, *layers:InputLayer|ConvolutionalLayer|PoolLayer):
        self.layers = layers
        if not isinstance(self.layers[0], InputLayer): raise TypeError("CNN must start with layer of type: InputLayer!")
        last_channel = self.layers[0].shape[-1]
        for cl in self.layers:
            if isinstance(cl, ConvolutionalLayer):
                cl.init_values(last_channel)
                last_channel = cl.filter_count


    def forward(self, input_array:NDArray) -> NDArray:
        if len(input_array.shape) == 3: input_array = input_array[None,]
        if input_array.shape[1:] != self.layers[0].shape:  # type: ignore
            raise TypeError(f"Input shape must match {self.layers[0].shape}") # type: ignore

        last_output = input_array
        for layer in self.layers:
            if isinstance(layer, ConvolutionalLayer):
                last_output = layer.forward(last_output)
            if isinstance(layer, PoolLayer):
                last_output = layer.forward(last_output)

        return last_output
    

    def backward(self, input_array:NDArray, y_true:NDArray, upstream:Optional[NDArray] = None, 
                 learning_rate = 1e-4) -> NDArray:
        if len(y_true.shape) == 3: y_true = y_true[None, ]
        y_pred = self.forward(input_array)
        if upstream: delta = upstream
        else: delta = np.sign(y_pred - y_true)

        for layer in self.layers[::-1]:
            if isinstance(layer, ConvolutionalLayer):
                delta = layer.backward(delta, learning_rate)

        return delta
    

    def save_network(self, file = "values.cnn"):
        with open(file, "wb") as f: pickle.dump(self.layers, f)


    def load_network(self, file = "values.cnn"):
        with open(file, "rb") as f: self.layers = pickle.load(f)



if __name__ == "__main__":
    cnn = CNN(
        InputLayer(16, 16, 3),
        ConvolutionalLayer(5, filter_shape=(4, 4), stride=4),
        ConvolutionalLayer(5, padding=3),
        ConvolutionalLayer(2, activation_function=ActivationFuncs.no_activation),
    )
    
    for _ in range(100):
        print(f"{_}/100%")
        cnn.backward(np.ones((16, 16, 3)), 
                     np.zeros((8, 8, 2)),
                     learning_rate = 1e-7)
        
    print(cnn.forward(np.ones((16, 16, 3))))
    