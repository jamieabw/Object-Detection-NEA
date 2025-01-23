import tkinter as tk
import tensorflow as tf
from PIL import Image
import random
import numpy as np
import os
from math import floor
from model.loss import YoloLoss
GPU = False
if GPU:
    physicalDevices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physicalDevices[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
N = 10 # 

class TrainingInfoHandler(tf.keras.callbacks.Callback):
    mapInputs = []
    mapPredictions = []
    mapTruths = []
    def __init__(self, guiInstance):
        super().__init__()
        self.batchCounter = 0
        self.currentEpoch = 0
        self.yoloLoss = None
        self.bboxLoss = None
        self.classLoss = None
        self.confidenceLoss = None
        self.guiInstance = guiInstance
        self.IoUThreshold = 0.5

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            totalSteps = self.params.get("steps")
            self.yoloLoss = round(logs.get('loss'), 3)
            self.confidenceLoss = round(logs.get('ConfidenceLoss'), 3)
            self.classLoss = round(logs.get('ClassLoss'), 3)
            self.bboxLoss = round(logs.get('boundingBoxLoss'), 3)
            self.setLossVals()
            self.updateEpochProgress(batch)
            self.batchCounter += 1

    def on_epoch_end(self, epoch, logs=None):
        self.batchCounter = 0
        self.currentEpoch += 1
        self.updateTrainingProgress()
        mAPDataHandler.mAPBatchPredict()
        self.mAP = self.calculateMAP()
        self.guiInstance.epochLossContainer[0].append(self.yoloLoss)
        self.guiInstance.epochLossContainer[1].append(self.confidenceLoss)
        self.guiInstance.epochLossContainer[2].append(self.classLoss)
        self.guiInstance.epochLossContainer[3].append(self.bboxLoss)
        self.guiInstance.mAPContainer.append(self.mAP)
        for i in range(len(self.recall)):
            self.guiInstance.recallContainer.append(self.recall[i])
            self.guiInstance.precisionContainer.append(self.precision[i])
        self.guiInstance.updatePlots()


    def updateEpochProgress(self, currentStep):
        totalSteps = self.params.get("steps")
        self.guiInstance.epochProgressBar["value"] = (currentStep / totalSteps) * 100

    def updateTrainingProgress(self):
        self.guiInstance.trainingProgressBar["value"] = (self.currentEpoch / self.guiInstance.totalEpochs) * 100

    def setLossVals(self):
        self.guiInstance.lossLabel.config(text=f"Current Overall Weighted Loss: {self.yoloLoss}")
        self.guiInstance.classLossLabel.config(text=f"Current Class Loss: {self.classLoss}")
        self.guiInstance.confidenceLossLabel.config(text=f"Current Confidence Loss: {self.confidenceLoss}")
        self.guiInstance.boundingBoxLossLabel.config(text=f"Current Bounding Box Loss: {self.bboxLoss}")
    
    def calculateIoU(self, predBox, trueBox):
        xTrue, yTrue, wTrue, hTrue = trueBox[1], trueBox[2], trueBox[3], trueBox[4] 
        xPred, yPred, wPred, hPred = predBox[1], predBox[2], predBox[3], predBox[4]
        # Convert center coordinates to corners
        x1Min, y1Min = xTrue - wTrue / 2, yTrue - hTrue / 2
        x1Max, y1Max = xTrue + wTrue / 2, yTrue + hTrue / 2
        
        x2Min, y2Min = xPred - wPred / 2, yPred - hPred / 2
        x2Max, y2Max = xPred + wPred / 2, yPred + hPred / 2
        
        # Calculate intersection coordinates
        interXmin = max(x1Min, x2Min)
        interYmin = max(y1Min, y2Min)
        interXmax = min(x1Max, x2Max)
        interYmax = min(y1Max, y2Max)
        
        # Calculate intersection area
        interWidth = max(0, interXmax - interXmin)
        interHeight = max(0, interYmax - interYmin)
        intersectionArea = interWidth * interHeight
        
        # Calculate areas of the boxes
        trueArea = (x1Max - x1Min) * (y1Max - y1Min)
        predArea = (x2Max - x2Min) * (y2Max - y2Min)
        unionArea = trueArea + predArea - intersectionArea
        if unionArea == 0:
            return 0.0
        
        iou = intersectionArea / unionArea
        return iou

    def calculateMAP(self):
        truePositives = 0
        falsePositives = 0
        falseNegatives = 0
        self.precision = []
        self.recall = []
        for a in range(len(TrainingInfoHandler.mapTruths)):
            for j in range(mAPDataHandler.model.BBOXES): # change to 7 if this breaks it
                for i in range(mAPDataHandler.model.BBOXES):
                    predictionCell = TrainingInfoHandler.mapPredictions[a][i][j]
                    trueCell = TrainingInfoHandler.mapTruths[a][i][j]
                    if predictionCell[0] < 0:
                        continue
                    IoU = self.calculateIoU(predictionCell, trueCell)
                    if IoU >= self.IoUThreshold:
                        if trueCell[0] == 1:
                            truePositives += 1
                        else:
                            falsePositives += 1
                    else:
                        if trueCell[0] == 1:
                            falseNegatives += 1
                    if truePositives + falsePositives == 0:
                        self.precision.append(0)
                    else:
                        self.precision.append(truePositives / (truePositives + falsePositives))
                    if truePositives + falseNegatives == 0:
                        self.recall.append(0)
                    else:
                        self.recall.append(truePositives / (truePositives + falseNegatives))
        self.recall = np.array(self.recall)
        self.precision = np.array(self.precision)
        sortedIndices = np.argsort(self.recall)
        self.recall = self.recall[sortedIndices]
        self.precision = self.precision[sortedIndices]
        # Iterate from second-to-last element to the first
        for i in range(len(self.precision)-2, -1, -1):
            self.precision[i] = max(self.precision[i], self.precision[i + 1])
        recallDelta = np.diff(self.recall, prepend=0)
        mAP = np.sum(self.precision * recallDelta)
        return mAP


            
class ModelTraining:
    def __init__(self, model, epochs, batchSize, learningRate, trainingDir, S, B, C, outputDir, guiInstance):
        self.model = model
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.trainingDir = trainingDir
        self.S = S
        self.B = B
        self.C = C
        self.outputDir = outputDir
        self.xTrain, self.yTrain = self.preprocessData()
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath= f"{self.outputDir}" + '\\modelSave_epoch_{epoch:02d}.h5',  # Use the built-in epoch variable
    save_weights_only=True,
    save_freq="epoch",
    verbose=1
)
        self.guiInstance = guiInstance
        loss = YoloLoss(B=self.B)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learningRate, momentum=0.9), loss=loss, metrics=["accuracy", loss.boundingBoxLoss, loss.ClassLoss, loss.ConfidenceLoss])
        
    def encodeLabels(self,textFileDir):
        label = np.zeros(shape=[self.S, self.S, (self.B * 5) + self.C])
        cellSize = 1 / self.S
        with open(textFileDir, "r") as t:
            for line in t.readlines():
                properties = line.split(" ")
                if len(properties) < 5:
                    continue  # Skip if the line does not contain enough properties
                
                # Extract bounding box and class info
                classId = int(properties[0])  # Class label
                bboxX = float(properties[1])  # X center
                bboxY = float(properties[2])  # Y center
                bboxW = float(properties[3])  # Width
                bboxH = float(properties[4])  # Height
                
                cellX = floor(bboxX // cellSize)  # Cell row index
                cellY = floor(bboxY // cellSize)  # Cell column index
                if bboxX > 1 or bboxY > 1:
                    continue
                
                # Calculate bounding box relative to the cell
                relativeX = (bboxX % cellSize) / cellSize
                relativeY = (bboxY % cellSize) / cellSize
                relativeW = bboxW / cellSize
                relativeH = bboxH / cellSize
                
                for b in range(self.B):
                    boxStart = (b * 5)

                    if label[cellX, cellY, boxStart] == 0:
                        label[cellX, cellY, boxStart] = 1
                        
                        label[cellX, cellY, boxStart + 1] = relativeX
                        label[cellX, cellY, boxStart + 2] = relativeY
                        label[cellX, cellY, boxStart + 3] = relativeW
                        label[cellX, cellY, boxStart + 4] = relativeH
                        
                        label[cellX, cellY, boxStart + 5 + classId] = 1
        # removal of nans which would break the loss
        return np.nan_to_num(label)
    

    def preprocessData(self, testData=False):
        images =[]
        labels =[] 
        counter = 0
        for file in os.listdir(self.trainingDir):
            if "txt" in file:
                labels.append(self.encodeLabels(f"{self.trainingDir}\\{file}"))
            else:
                counter +=1
                if testData:
                    images.append(f"{self.trainingDir}\\{file}")
                else:    
                    images.append(f"{self.trainingDir}\\{file}")
        
        images = np.array(images)
        labels = np.array(labels)
        print(labels.shape)
        # test data cannot be shuffled here, as some test data will not have a corresponding annotation
        # therefore elements would be mismatched in both arrays leading to problems when testing the model
        if not testData:
            indices = np.arange(len(images))
            np.random.shuffle(indices)
            shuffledImages = images[indices]
            print(labels.shape)
            shuffledLabels = labels[indices, ...]
            return shuffledImages, shuffledLabels

    def dataGenerator(self):
        startVal = 0
        imagePaths = list(self.xTrain)
        labels = list(self.yTrain)

        while True:
            if startVal == 0:  # Shuffle at the start of each epoch
                combined = list(zip(imagePaths, labels))
                random.shuffle(combined)
                imagePaths, labels = zip(*combined)

            endVal = min(startVal + self.batchSize, len(labels))
            batchInput = []
            batchOutput = []
            for i in range(startVal, endVal):
                batchInput.append(convertToArray(imagePaths[i]))
                batchOutput.append(labels[i])
            
            startVal += self.batchSize
            if startVal >= len(labels):
                startVal = 0
            if endVal == len(labels):
                    mAPDataHandler.gatherMapData(batchInput, batchOutput)
                    mAPDataHandler.model = self.model
            yield (np.array(batchInput).astype("float32") / 255.0, np.array(batchOutput))

    def train(self):
        self.model.build((None, 448,448,3))
        self.model.fit(self.dataGenerator(),
                        epochs=self.epochs, verbose=1, steps_per_epoch=len(self.yTrain) /self.batchSize,
                          callbacks=[self.checkpoint, TrainingInfoHandler(self.guiInstance)])
        

class mAPDataHandler:
    model = None
    def gatherMapData(inputBatch, outputBatch):
        TrainingInfoHandler.mapTruths = outputBatch
        TrainingInfoHandler.mapInputs = inputBatch
            
    @classmethod
    def mAPBatchPredict(cls):
        inputBatch = np.array(TrainingInfoHandler.mapInputs)
        outputBatch = np.array(TrainingInfoHandler.mapTruths)
        predictions = []
        print(inputBatch.shape)
        for step in inputBatch:
            x = np.array(step).reshape((1,448,448,3)).astype("float32") / 255.0
            predictions.append(cls.model.predict(x)[0])
        TrainingInfoHandler.mapPredictions = predictions

def convertToArray(imagePath, size=(448,448)):
    image = Image.open(imagePath)
    return np.transpose(np.array(image.resize(size)), (1,0,2))[:, :, :3] # np.array turns the image into an array of [h,w,c] it needs to be [w,h,c]

# converts a numpy array into a PIL image and matches colour channels correctly
def convertToImage(image):
    image *= 255
    image = np.array(image, dtype="uint8")
    return Image.fromarray(np.transpose(image, (1,0,2)))



    
