
import tensorflow as tf
from keras import layers
from CNNblocks import CNNBlock
from loss import YoloLoss#, boundingBoxLoss, ClassLoss, ConfidenceLoss
from predictionHandler import findBoxes
from dataProcessing import preprocessData, convertToArray
import numpy as np
import random
# DEBUGGING
GPU = not True
if GPU:
    physicalDevices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physicalDevices[0], True)
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
lrSchedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([200,400,600,20000,30000], [0.00035 * scale for scale in [2.5,2,1.5,1,0.5,0.1]])
startVal = 0
l2Regularizer = tf.keras.regularizers.l2(0)
trainingDirectory = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\training data set"
validationDirectory = 'C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\PASCAL VOC\\valid\\VOCdevkit\\VOC2007\\JPEGImages'
testDirectory = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\dataset-humans\\INRIA Person detection dataset.v1i.darknet\\test"
"C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\dataset-humans\\INRIA Person detection dataset.v1i.darknet\\test"
"C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\knives\\test"

#TODO: CLEAN THIS CODE UP ITS A MESS BUT DONT BREAK ANYTHING, IMPLEMENT NMS AND THEN ALSO TEST IT ON A UNSEEN IMAGE

# TEST DATA REMOVE AFTER
xTrain, yTrain =preprocessData(trainingDirectory, GRID_SIZE, BBOXES, CLASSES)
xValid, yValid = preprocessData(validationDirectory, GRID_SIZE, BBOXES, CLASSES)
xTest, yTest =preprocessData(testDirectory, GRID_SIZE, BBOXES, CLASSES, True)

def dataGenerator(imagePaths, labels, batchSize):
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
    def __init__(self, gridSize=GRID_SIZE, classes=CLASSES, bboxes=BBOXES):
        super(YoloV1, self).__init__()
        self.gridSize = gridSize
        
        self.classes = classes
        self.bboxes = bboxes
        self.convLayers = CNNBlock(KERNELS, SIZES, STRIDES, l2Regularizer)
        self.denseLayer = layers.Dense(4096, kernel_regularizer=l2Regularizer)
        self.outputDense = layers.Dense(self.gridSize * self.gridSize * ((5 * self.bboxes) + self.classes), kernel_regularizer=l2Regularizer) # adding a relu activation to this breaks the loss i think - dunno why

    def call(self, inputs):
        x = self.convLayers(inputs)
        x = layers.Flatten()(x)
        x = self.denseLayer(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        #x = layers.Dropout(0.5)(x)
        x = self.outputDense(x)
        x =  layers.Reshape((self.gridSize, self.gridSize, (5 * self.bboxes) + self.classes))(x)
        return x

    
"""model = YoloV1()#testModel(num_classes=1, num_boxes=1)#YoloV1()
model.build(input_shape=(None, 448, 448, 3))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lrSchedule, momentum=0.9), loss=yoloLoss, metrics=["accuracy", boundingBoxLoss, ClassLoss, ConfidenceLoss])
#CHANGE THIS BNACK TO LEARNING SCHEDULE ASAP
model.summary()
print(yTrain.shape)
print(yTest.shape)
# Load the weights
model.load_weights("E:\\IMPORTANT MODEL SAVES FOR NEA\\YOLOV1_v5.h5")
#model.save_weights("E:\\IMPORTANT MODEL SAVES FOR NEA\\YOLOV1_v4.h5")""" 



def train():
    model.fit(dataGenerator(xTrain, yTrain,12), epochs=40, verbose=1, steps_per_epoch=len(yTrain) /12, callbacks=[checkpoint],
            validation_data=dataGenerator(xValid, yValid, 12), validation_steps=len(yValid) / 12)#, validation_data=data_generator(x_valid, y_valid,16),validation_steps=len(y_valid) / 16)"""

def test():
    for data in xTest:
        print(data.shape)
        data = convertToArray(data).astype("float32") / 255.0
        print(data.shape)
        data2 = np.reshape(data, (1,448,448,3))
        print(data.shape)
        findBoxes(data, model.predict(data2))




def calculatemAP(predictions, truth):
    TP = 0
    FP = 0
    FN = 0
    precision = []
    recall = []
    for j in range(7):
        for i in range(7):
            predictionCell = predictions[i][j]
            trueCell = truth[i][j]
            if predictionCell[0] < 0:
                continue
            IoU = calculateIoU(predictionCell, trueCell)
            if IoU >= IoUThreshold:
                if trueCell[0] == 1:
                    TP += 1
                
                else:
                    FP += 1
            else:
                if trueCell[0] == 1:
                    FN += 1
            if TP + FP == 0:
                precision.append(0)
            else:
                precision.append(TP / (TP + FP))
            if TP + FN == 0:
                recall.append(0)
            else:
                recall.append(TP / (TP + FN))
    print(precision)
    print(recall)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='o')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.show()



def calculateIoU(predBox, trueBox):
    conTrue, xTrue, yTrue, wTrue, hTrue, classTrue = trueBox
    conPred, xPred, yPred, wPred, hPred, classPred = predBox

    # Calculate corners of the bounding boxes
    topLeftTrue = [xTrue - (wTrue / 2), yTrue - (hTrue / 2)]
    bottomRightTrue = [xTrue + (wTrue / 2), yTrue + (hTrue / 2)]
    topLeftPred = [xPred - (wPred / 2), yPred - (hPred / 2)]
    bottomRightPred = [xPred + (wPred / 2), yPred + (hPred / 2)]

    # Calculate intersection rectangle
    xLeft = max(topLeftTrue[0], topLeftPred[0])
    yTop = max(topLeftTrue[1], topLeftPred[1])
    xRight = min(bottomRightTrue[0], bottomRightPred[0])
    yBottom = min(bottomRightTrue[1], bottomRightPred[1])

    # Check for no intersection
    if xRight <= xLeft or yBottom <= yTop:
        return 0.0

    # Calculate intersection area
    intersection = (xRight - xLeft) * (yBottom - yTop)

    # Calculate areas of the individual boxes
    areaTrue = wTrue * hTrue
    areaPred = wPred * hPred

    # Calculate union area
    union = areaTrue + areaPred - intersection

    # Calculate IoU
    iou = intersection / union
    return iou

if __name__ == "__main__":
    model = YoloV1()#testModel(num_classes=1, num_boxes=1)#YoloV1()
    model.build(input_shape=(None, 448, 448, 3))
    loss = YoloLoss()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lrSchedule, momentum=0.9), loss=loss, metrics=["accuracy", loss.boundingBoxLoss, loss.ClassLoss, loss.ConfidenceLoss])
    #CHANGE THIS BNACK TO LEARNING SCHEDULE ASAP
    model.summary()
    print(yTrain.shape)
    print(yTest.shape)
    # Load the weights
    #model.load_weights("C:\\Users\\jamie\\Desktop\\saVES\\YOLOV1_v5.h5")
    model.save_weights("E:\\IMPORTANT MODEL SAVES FOR NEA\\YOLOV1_v5.h5") 
    #train()
    print(xTrain.shape)
    print(xTrain)
    for i, data in enumerate(xTrain):
        truth = yTrain[i]
        print(data.shape)
        print(truth.shape)
        prediction = model.predict(data)
        print(calculatemAP(prediction, truth))



IoUThreshold=0.4
import matplotlib.pyplot as plt





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
        print(lossTestTrue



"""
