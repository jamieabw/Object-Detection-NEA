import tensorflow as tf
from keras import layers
from CNNblocks import CNNBlock
from loss import yoloLoss
from predictionHandler import findBoxes
from dataProcessing import preprocessData, convertToArray
import numpy as np
# DEBUGGING
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TODO: THE LOSS FUNCTION IS FUCKED I BELIEVE, WHICH IS CAUSING THE OVERFITTING SO TRY FIX THAT, ALSO ADD THE DROPOUT AND EXTRA THINGS MENTIONED 
# IN THE YOLO PAPER TO ASSIST WITH TRAINING AS MAYBE THAT IS THE PROBLEM!!!!!

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'./modelsaves\\weights.h5', save_weights_only=True, save_freq="epoch", verbose=1)

# Define the architecture parameters
KERNELS = [64, 192, 128, 256, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 512, 1024, 512, 1024, 512, 1024, 1024, 1024]
SIZES = [7, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3]
STRIDES = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
GRID_SIZE = 8
CLASSES = 1 # training only on crowdhuman initially
BBOXES = 1
startVal = 0
trainingDirectory = "C:\\Users\\jamie\Documents\\CS NEA 24 25 source code\\datasets\\morehumans\\train"
testDirectory = "C:\\Users\\jamie\Documents\\CS NEA 24 25 source code\\datasets\\dataset-humans\\INRIA Person detection dataset.v1i.darknet\\test"


#TODO: CLEAN THIS CODE UP ITS A MESS BUT DONT BREAK ANYTHING, IMPLEMENT NMS AND THEN ALSO TEST IT ON A UNSEEN IMAGE

# TEST DATA REMOVE AFTER
x_train, y_train =preprocessData(trainingDirectory)
x_test, y_test =preprocessData(testDirectory)

def data_generator(imagePaths, labels, batchSize):
    startVal = 0

    while True:
        endVal = min(startVal + batchSize, len(labels))
        batchInput = []
        batchOutput = []
        for i in range(startVal, endVal):
            batchInput.append(convertToArray(imagePaths[i]))
            batchOutput.append(labels[i])
        startVal += batchSize
        if startVal >= len(labels):
            startVal = 0 

        yield (np.array(batchInput).astype("float32") / 255.0, np.array(batchOutput))


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
print(y_train.shape)
model.fit(data_generator(x_train, y_train, 16), epochs=1, verbose=1, steps_per_epoch=len(y_train) / 16, callbacks=[checkpoint])
for data in x_test:
    data = convertToArray(data).astype("float32") / 255.0
    print(data.shape)
    data2 = np.reshape(data, (1,448,448,3))
    print(data.shape)
    findBoxes(data, model.predict(data2))  

"""NOTE: IF THE TRAINING IS EXTREMELY FUCKING SLOW, ONLY USE ONE MONITOR IT SEEMS TO DO THE TRICK"""