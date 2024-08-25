
import tensorflow as tf
import keras
from keras import layers

POOLING_POSITIONS = [1, 5, 15]


class CNNBlock(layers.Layer):
    def __init__(self, kernels, sizes, strides):
        super(CNNBlock, self).__init__()
        self.convLayers = [layers.Conv2D(k, a, s, activation="leaky_relu", padding="same") for k, a, s in zip(kernels, sizes, strides)]
        self.poolingLayers = [layers.MaxPooling2D(2, 2, padding="same") for i in range(4)]

    def call(self, inputs):
        x = self.convLayers[0](inputs)
        x = self.poolingLayers[0](x)
        for j in range(1, len(self.convLayers)):
            #print(f"layer {j + 1}")
            x = self.convLayers[j](x)
            if j in POOLING_POSITIONS:
                #print(f"pooling layer after layer {j + 1}")
                index = POOLING_POSITIONS.index(j)
                x = self.poolingLayers[index](x)
        return x
    