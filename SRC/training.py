import tkinter as tk
import tensorflow as tf
from PIL import Image
import random
import numpy as np
import os
from math import floor
from model.loss import YoloLoss
GPU = True
if GPU:
    physicalDevices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physicalDevices[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
"""
this file will contain another class for another GUI which has the sole purpose of defining new models and training them, this GUI will have toplevels which display
graphs of the various infomation of the training process aswell as displaying the values of the losses and the mAP during training
"""

""" need to figure out how to get the data :( )"""
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
            """if self.batchCounter > totalSteps - N:
                #batchData = logs.get("batch_data")
                #print(batchData)
                images, labels = self.getBatch(batch)
                for i in range(len(images)):
                    self.predictionsForMAP.append(self.guiInstance.master.model.predict(images[i]))
                    self.truthsForMAP.append(labels[i])"""
            self.batchCounter += 1

    def on_epoch_end(self, epoch, logs=None):
        self.batchCounter = 0
        self.currentEpoch += 1
        self.updateTrainingProgress()
        print(self.yoloLoss, self.bboxLoss)
        #self.guiInstance.epochLossContainer.append((self.yoloLoss, self.confidenceLoss, self.classLoss, self.bboxLoss))
        self.guiInstance.epochLossContainer[0].append(self.yoloLoss)
        self.guiInstance.epochLossContainer[1].append(self.confidenceLoss)
        self.guiInstance.epochLossContainer[2].append(self.classLoss)
        self.guiInstance.epochLossContainer[3].append(self.bboxLoss)
        self.guiInstance.updatePlot()
        mAPDataHandler.mAPBatchPredict()
        self.calculateMAP()


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

    """def getBatch(self, batch):
        print(self.params)
        images, labels = self.params.get("train_data")
        return images, labels"""
    
    def calculateIoU(self, predBox, trueBox):
        xTrue, yTrue, wTrue, hTrue = trueBox[1], trueBox[2], trueBox[3], trueBox[4] 
        xPred, yPred, wPred, hPred = predBox[1], predBox[2], predBox[3], predBox[4]

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

    def calculateMAP(self):
        """
        THIS IS WHAT I AM CURRENTLY WORKING ON.
        TODO: everything works except from these values, they arent being generated properly, just a list of 0s
        so will need to fix that and implement the final two graphs and everything is finished wheyy
        """
        TP = 0
        FP = 0
        FN = 0
        precision = []
        recall = []
        print(np.array(TrainingInfoHandler.mapPredictions).shape)
        print(np.array(TrainingInfoHandler.mapTruths).shape)
        for a in range(len(TrainingInfoHandler.mapTruths)):
            for j in range(7):
                for i in range(7):
                    predictionCell = TrainingInfoHandler.mapPredictions[a][i][j]
                    trueCell = TrainingInfoHandler.mapTruths[a][i][j]
                    if predictionCell[0] < 0:
                        continue
                    IoU = self.calculateIoU(predictionCell, trueCell)
                    if IoU >= self.IoUThreshold:
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
                    #print(recall)
                    #print(precision)
            
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
        print(f"FUCKING USELESS B: {self.B}")
        print(f"FUCKING USELESS C: {self.C}")
        print(f"FUCKING USELESS S: {self.S}")
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
        # Create label array: [S, S, B * (5 + C)]
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
                
                # Determine which cell the bounding box belongs to
                cellX = floor(bboxX // cellSize)  # Cell row index
                cellY = floor(bboxY // cellSize)  # Cell column index
                if bboxX > 1 or bboxY > 1:
                    continue
                
                # Calculate bounding box relative to the cell
                relativeX = (bboxX % cellSize) / cellSize
                relativeY = (bboxY % cellSize) / cellSize
                relativeW = bboxW / cellSize
                relativeH = bboxH / cellSize
                
                # Assign values for each bounding box (support multiple B)
                for b in range(self.B):
                    boxStart = (b * 5)  # Start index for this box in the label array
                    
                    # If the confidence is 0 (no bounding box assigned yet), assign this box
                    #print(cell_x, cell_y, bbox_y)
                    if label[cellX, cellY, boxStart] == 0:
                        # Assign the confidence (objectness)
                        label[cellX, cellY, boxStart] = 1
                        
                        # Encode the bounding box (x, y, w, h)
                        label[cellX, cellY, boxStart + 1] = relativeX
                        label[cellX, cellY, boxStart + 2] = relativeY
                        label[cellX, cellY, boxStart + 3] = relativeW
                        label[cellX, cellY, boxStart + 4] = relativeH
                        
                        # One-hot encode the class (start from box_start + 5)
                        label[cellX, cellY, boxStart + 5 + classId] = 1
                        
                
        return np.nan_to_num(label)
    

    def preprocessData(self, testData=False):
        images =[] #np.array([])
        labels =[] #np.array([])
        counter = 0
        for file in os.listdir(self.trainingDir):
            if "txt" in file:
                labels.append(self.encodeLabels(f"{self.trainingDir}\\{file}"))
                #np.append(labels, encodeLabels(f"{folderDir}\\{file}", 8,1,1))
            else:
                counter +=1
                if testData:
                    ...#convertToArray(images.append(f"{folderDir}\\{file}"))
                    images.append(f"{self.trainingDir}\\{file}")
                else:    
                    images.append(f"{self.trainingDir}\\{file}")
                #np.append(images, jpg_to_resized_array(f"{folderDir}\\{file}")

        
        images = np.array(images)
        labels = np.array(labels)
        print(labels.shape)
        # test data cannot be shuffled here, as some test data will not have a corresponding annotation
        # therefore elements would be mismatched in both arrays leading to problems when testing the model
        if not testData:
            indices = np.arange(len(images))
            np.random.shuffle(indices)

            # Apply the permutation to both arrays
            shuffledImages = images[indices]
            print(labels.shape)
            shuffledLabels = labels[indices, ...]
            return shuffledImages, shuffledLabels

    def dataGenerator(self):
        startVal = 0

        # Convert imagePaths and labels to lists to allow shuffling
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
        #print(outputBatch.shape)
        print(inputBatch.shape)
        for step in inputBatch:
            
            #print(step.shape)
            predictions.append(cls.model.predict(np.array(step).reshape((1,448,448,3)).astype("float32") / 255.0)[0])
        TrainingInfoHandler.mapPredictions = predictions

def convertToArray(imagePath, size=(448,448)):
    image = Image.open(imagePath)
    return np.transpose(np.array(image.resize(size)), (1,0,2))[:, :, :3] # np.array turns the image into an array of [h,w,c] it needs to be [w,h,c]

# converts a numpy array into a PIL image and matches colour channels correctly
def convertToImage(image):
    image *= 255
    image = np.array(image, dtype="uint8")
    return Image.fromarray(np.transpose(image, (1,0,2)))



    
