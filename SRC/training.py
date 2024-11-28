import tkinter as tk
import tensorflow as tf
from PIL import Image
import random
import numpy as np
import os
from math import floor
from model.loss import boundingBoxLoss, ClassLoss, ConfidenceLoss, yoloLoss
GPU = not True
if GPU:
    physicalDevices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physicalDevices[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
"""
this file will contain another class for another GUI which has the sole purpose of defining new models and training them, this GUI will have toplevels which display
graphs of the various infomation of the training process aswell as displaying the values of the losses and the mAP during training
"""



class TrainingInfoHandler(tf.keras.callbacks.Callback):
    def __init__(self, guiInstance):
        super().__init__()
        self.currentEpoch = 0
        self.yoloLoss = None
        self.bboxLoss = None
        self.classLoss = None
        self.confidenceLoss = None
        self.guiInstance = guiInstance

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            self.currentEpoch += 1
            self.yoloLoss = round(logs.get('loss'), 3)
            self.confidenceLoss = round(logs.get('ConfidenceLoss'), 3)
            self.classLoss = round(logs.get('ClassLoss'), 3)
            self.bboxLoss = round(logs.get('boundingBoxLoss'), 3)
            self.setLossVals()
            self.updateEpochProgress(batch)

    def on_epoch_end(self, epoch, logs=None):
        self.updateTrainingProgress()


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
    filepath='{self.outputDir}\\modelSave_epoch_{epoch:02d}.h5',  # Use the built-in epoch variable
    save_weights_only=True,
    save_freq="epoch",
    verbose=1
)
        self.guiInstance = guiInstance
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learningRate, momentum=0.9), loss=yoloLoss, metrics=["accuracy", boundingBoxLoss, ClassLoss, ConfidenceLoss])
        
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
                    boxStart = b * (5 + self.C)  # Start index for this box in the label array
                    
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
                        
                        break  # Only assign one bounding box per object
                
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
            
            yield (np.array(batchInput).astype("float32") / 255.0, np.array(batchOutput))

    def train(self):
        self.model.fit(self.dataGenerator(),
                        epochs=self.epochs, verbose=1, steps_per_epoch=len(self.yTrain) /self.batchSize,
                          callbacks=[self.checkpoint, TrainingInfoHandler(self.guiInstance)])
        

def convertToArray(imagePath, size=(448,448)):
    image = Image.open(imagePath)
    return np.transpose(np.array(image.resize(size)), (1,0,2))[:, :, :3] # np.array turns the image into an array of [h,w,c] it needs to be [w,h,c]

# converts a numpy array into a PIL image and matches colour channels correctly
def convertToImage(image):
    image *= 255
    image = np.array(image, dtype="uint8")
    return Image.fromarray(np.transpose(image, (1,0,2)))



    
