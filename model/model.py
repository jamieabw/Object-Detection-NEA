import tensorflow as tf
from keras import layers
from CNNblocks import CNNBlock
from loss import yoloLoss, boundingBoxLoss, ClassLoss, ConfidenceLoss
from predictionHandler import findBoxes
from dataProcessing import preprocessData, convertToArray
import numpy as np
import random
# DEBUGGING
GPU = True
if GPU:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TODO: THE LOSS FUNCTION IS FUCKED I BELIEVE, WHICH IS CAUSING THE OVERFITTING SO TRY FIX THAT, ALSO ADD THE DROPOUT AND EXTRA THINGS MENTIONED 
# IN THE YOLO PAPER TO ASSIST WITH TRAINING AS MAYBE THAT IS THE PROBLEM!!!!!

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='C:\\Users\\jamie\\Desktop\\saVES\\modelSave_epoch_{epoch:02d}.h5',  # Use the built-in epoch variable
    save_weights_only=True,
    save_freq="epoch",
    verbose=1
)

# Define the architecture parameters
KERNELS = [64, 192, 128, 256, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 512, 1024, 512, 1024, 512, 1024, 1024, 1024]
SIZES = [7, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3]
STRIDES = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
GRID_SIZE = 7
CLASSES = 1 # training only on crowdhuman initially
BBOXES = 1
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([200,400,600,20000,30000], [0.00035 * scale for scale in [2.5,2,1.5,1,0.5,0.1]])
startVal = 0
l2_regularizer = tf.keras.regularizers.l2(0)
trainingDirectory = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages"
validationDirectory = 'C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\PASCAL VOC\\valid\\VOCdevkit\\VOC2007\\JPEGImages'
testDirectory = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\dataset-humans\\INRIA Person detection dataset.v1i.darknet\\test"
"C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\dataset-humans\\INRIA Person detection dataset.v1i.darknet\\test"
"C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\knives\\test"

#TODO: CLEAN THIS CODE UP ITS A MESS BUT DONT BREAK ANYTHING, IMPLEMENT NMS AND THEN ALSO TEST IT ON A UNSEEN IMAGE

# TEST DATA REMOVE AFTER
x_train, y_train =preprocessData(trainingDirectory, GRID_SIZE, BBOXES, CLASSES)
x_valid, y_valid = preprocessData(validationDirectory, GRID_SIZE, BBOXES, CLASSES)
x_test, y_test =preprocessData(testDirectory, GRID_SIZE, BBOXES, CLASSES, True)

def data_generator(imagePaths, labels, batchSize):
    startVal = 0

    # Convert imagePaths and labels to lists to allow shuffling
    imagePaths = list(imagePaths)
    labels = list(labels)

    while True:
        if startVal == 0:  # Shuffle at the start of each epoch
            combined = list(zip(imagePaths, labels))
            random.shuffle(combined)
            imagePaths, labels = zip(*combined)

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
        self.convLayers = CNNBlock(KERNELS, SIZES, STRIDES, l2_regularizer)
        self.denseLayer = layers.Dense(4096, kernel_regularizer=l2_regularizer)
        self.outputDense = layers.Dense(self.grid_size * self.grid_size * ((5 * self.bboxes) + self.classes), kernel_regularizer=l2_regularizer) # adding a relu activation to this breaks the loss i think - dunno why

    def call(self, inputs):
        x = self.convLayers(inputs)
        x = layers.Flatten()(x)
        x = self.denseLayer(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        #x = layers.Dropout(0.5)(x)
        x = self.outputDense(x)
        x =  layers.Reshape((self.grid_size, self.grid_size, (5 * self.bboxes) + self.classes))(x)
        return x

    
model = YoloV1()#testModel(num_classes=1, num_boxes=1)#YoloV1()
model.build(input_shape=(None, 448, 448, 3))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9), loss=yoloLoss, metrics=["accuracy", boundingBoxLoss, ClassLoss, ConfidenceLoss])
#CHANGE THIS BNACK TO LEARNING SCHEDULE ASAP
model.summary()
print(y_train.shape)
print(y_test.shape)
# Load the weights
model.load_weights("C:\\Users\\jamie\\Desktop\\saVES\\modelSave_epoch_20.h5")
model.save_weights("E:\\IMPORTANT MODEL SAVES FOR NEA\\YOLOV1_v4.h5") 



model.fit(data_generator(x_train, y_train,12), epochs=40, verbose=1, steps_per_epoch=len(y_train) /12, callbacks=[checkpoint],
           validation_data=data_generator(x_valid, y_valid, 12), validation_steps=len(y_valid) / 12)#, validation_data=data_generator(x_valid, y_valid,16),validation_steps=len(y_valid) / 16)"""
"""lossTestData = convertToArray(x_test[1]).astype("float32") / 255.0
lossTestTrue = y_test[1].reshape((1,8,8,25))
print(lossTestData.shape)
while True:
    model.fit(np.reshape(lossTestData, (1,448,448,3)), lossTestTrue, verbose=1, epochs=220)
    result = model.predict(np.reshape(lossTestData, (1,448,448,3)))
    findBoxes(lossTestData, result)
    print("\n\n\n\n")

        print("\n\n\n\n")
        print(result)
        print("\n\n\n\n")
        print(lossTestTrue)


def test():
    for data in x_test:
        print(data.shape)
        data = convertToArray(data).astype("float32") / 255.0
        print(data.shape)
        data2 = np.reshape(data, (1,448,448,3))
        print(data.shape)
        findBoxes(data, model.predict(data2))


if __name__ == "__main__":
    model = YoloV1()#testModel(num_classes=1, num_boxes=1)#YoloV1()
    model.build(input_shape=(None, 448, 448, 3))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9), loss=yoloLoss, metrics=["accuracy", boundingBoxLoss, ClassLoss, ConfidenceLoss])
    #CHANGE THIS BNACK TO LEARNING SCHEDULE ASAP
    model.summary()
    print(y_train.shape)
    print(y_test.shape)
    # Load the weights
    model.load_weights("C:\\Users\\jamie\\Desktop\\saVES\\YOLOV1_v5.h5")
    #model.save_weights("E:\\IMPORTANT MODEL SAVES FOR NEA\\YOLOV1_v5.h5") 
    test()


"""NOTE AND TODO: right there is an issue with training, need to try a way bigger dataset and try it on that to see if it breaks the network like
THE ISSUE SEEMS TO BE WITH 0KB TXT FILES WHERE THERE ARE NO OBJETS """


"""FINAL SUMMER NOTE --- the model seems to be able to learn specific datasets decently well (such as the drones) however fails to learn of others (predicts
 basically the same bounding box no matter the image), need to add a way to convert other YOLO forms into YOLO darknet form so it can be learnt,
also need to implement mAP calculation and things like IOU and NMS, if the model doesnt learn larger datasets well enough then a full recreation of the project
could be a solution to ensure the code is actually right and that i havent fucked something random up somewhere. once the training is working properly the rest of
the project should be quite easy to finish."""



""" DO THIS NEXT BELOW"""
# NOTE: a way to test the loss function will be to continuously train it on one piece of test data to set if it actually improves!! do this when possible

"""NOTE: IF THE TRAINING IS EXTREMELY FUCKING SLOW, ONLY USE ONE MONITOR IT SEEMS TO DO THE TRICK"""

"""it is training, i now need to get it so it can import multiple classes not just one and train it on that"""