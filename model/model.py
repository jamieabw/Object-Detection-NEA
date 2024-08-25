import tensorflow as tf
from keras import layers
from CNNblocks import CNNBlock
import numpy as np
from loss import yoloLoss
# DEBUGGING
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

"""
TODO: sort out the output labels, make a few conversion files, test the model on a few bits of data then start training
"""

# Define the architecture parameters
KERNELS = [64, 192, 128, 256, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 512, 1024, 512, 1024, 512, 1024, 1024, 1024]
SIZES = [7, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3]
STRIDES = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
GRID_SIZE = 9
CLASSES = 1 # training only on crowdhuman initially
BBOXES = 1

class YoloV1(tf.keras.Model):
    def __init__(self, grid_size=GRID_SIZE, classes=CLASSES, bboxes=BBOXES):
        super(YoloV1, self).__init__()
        self.grid_size = grid_size
        self.classes = classes
        self.bboxes = bboxes
        self.convLayers = CNNBlock(KERNELS, SIZES, STRIDES)
        self.denseLayer = layers.Dense(4096)
        self.outputDense = layers.Dense(self.grid_size * self.grid_size * ((5 * self.bboxes) + self.classes))

    def call(self, inputs):
        x = self.convLayers(inputs)
        print(f"Shape after CNNBlock: {x.shape}")
        x = layers.Flatten()(x)
        x = self.denseLayer(x)
        print(f"Shape after denmse: {x.shape}")
        x = self.outputDense(x)
        print(f"Shape after dense: {x.shape}")
        x =  layers.Reshape((self.grid_size, self.grid_size, (5 * self.bboxes) + self.classes))(x)
        return x    
    
model = YoloV1()
model.build(input_shape=(None, 448, 448, 3))
# Summary of the model
"""
When using a custom loss function in Keras with model.compile, you typically need to pass a function that takes two arguments: yTrue and yPred.
 Keras will automatically pass the true labels and predicted values to this loss function during training.
 In your case, your yoloLoss function is already designed to take these arguments, so you don't need to modify it further."""
model.compile(optimizer="adam", loss=yoloLoss, metrics=["accuracy"])
model.summary()