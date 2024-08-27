import tensorflow as tf
from keras import layers
from CNNblocks import CNNBlock
from loss import yoloLoss
from predictionHandler import findBoxes
from dataProcessing import preprocessData
import numpy as np
# DEBUGGING
from PLEASEJUSTWORK import results
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



# Define the architecture parameters
KERNELS = [64, 192, 128, 256, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 512, 1024, 512, 1024, 512, 1024, 1024, 1024]
SIZES = [7, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3]
STRIDES = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
GRID_SIZE = 8
CLASSES = 1 # training only on crowdhuman initially
BBOXES = 1
trainingDirectory = "C:\\Users\\jamie\Documents\\CS NEA 24 25 source code\\datasets\\morehumans\\train"
testDirectory = "C:\\Users\\jamie\Documents\\CS NEA 24 25 source code\\datasets\\dataset-humans\\INRIA Person detection dataset.v1i.darknet\\test"

#TODO: CLEAN THIS CODE UP ITS A MESS BUT DONT BREAK ANYTHING, IMPLEMENT NMS AND THEN ALSO TEST IT ON A UNSEEN IMAGE

# TEST DATA REMOVE AFTER
x_train, y_train = preprocessData(trainingDirectory)
x_test, y_test = preprocessData(testDirectory)

class YoloV1(tf.keras.Model):
    def __init__(self, grid_size=GRID_SIZE, classes=CLASSES, bboxes=BBOXES):
        super(YoloV1, self).__init__()
        self.grid_size = grid_size
        self.classes = classes
        self.bboxes = bboxes
        self.convLayers = CNNBlock(KERNELS, SIZES, STRIDES)
        self.denseLayer = layers.Dense(4096)
        self.outputDense = layers.Dense(self.grid_size * self.grid_size * ((5 * self.bboxes) + self.classes)) # adding a relu activation to this breaks the loss i think - dunno why

    def call(self, inputs):
        x = self.convLayers(inputs)
        x = layers.Flatten()(x)
        x = self.denseLayer(x)
        x = self.outputDense(x)
        x =  layers.Reshape((self.grid_size, self.grid_size, (5 * self.bboxes) + self.classes))(x)
        return x    
    
model = YoloV1()
model.build(input_shape=(None, 448, 448, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=yoloLoss, metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, epochs=4, verbose=1, batch_size=3)
for data in x_test:
    data2 = np.reshape(data, (1,448,448,3))
    #print(x_test.shape)
    print(data.shape)
    findBoxes(data, model.predict(data2))  

"""NOTE: IF THE TRAINING IS EXTREMELY FUCKING SLOW, ONLY USE ONE MONITOR IT SEEMS TO DO THE TRICK"""