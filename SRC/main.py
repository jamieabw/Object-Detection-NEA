
import tkinter as tk
from tkinter import colorchooser
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import keras.regularizers
from videoHandler import yieldNextFrame, readImage
import tensorflow as tf
from keras.models import load_model
from keras import layers
import numpy as np
from model.CNNblocks import CNNBlock
#from model.model import YoloV1# this wont work for some reason, something to do with CNNBlocks
import keras
import subprocess
import cv2
import threading
import time
import traceback
from training import ModelTraining
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
DEFAULT_THRESHOLD = 0.4
CLASSES = 1
GRID_SIZE = 7
BBOXES = 1
KERNELS = [64, 192, 128, 256, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 512, 1024, 512, 1024, 512, 1024, 1024, 1024]
SIZES = [7, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3]
STRIDES = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
l2Regularizer = keras.regularizers.l2(0)
DEFAULT_CAM_SOURCE = 0
DEFAULT_BBOX_COLOUR = "#FF0000"
DEFAULT_BBOX_WIDTH = 2
DEFAULT_MODEL_PATH = "E:\\IMPORTANT MODEL SAVES FOR NEA\\YOLOV1_v5.h5"

"""
TODO:
- add a function to calculate mAP
- add a window which displays statistical info about the training process, include: losses, mAP, epoch number, maybe a preview window
which displays the results of running the model on a unique not seen before image after each epoch
- add the choice of a penalising l2 regulariser
- add support to use different GPUs for training
- add a debug menu (this could display, current FPS)
- train a model to sufficiently and effectively make detections on multiple objects
- ensure all functions are made by ME, ensure all AI generations and support from websites is eliminated
- sufficiently comment the code so a third party programmers understands why specific implementations are how they are
"""

# a place holder function used when loading an ENTIRE MODEL AND NOT ONLY WEIGHTS
# this is no longer applicable due to the use of subclassing in a model which 
# breaks loading an entire model
def yoloLossPlaceholder():
    pass

class webcamThreadHandler:
    webcams = {}

    @classmethod
    def getWebcamDevices(cls):
        result = subprocess.run(['wmic', 'path', 'Win32_PnPEntity', 'where', 'Caption like "%cam%"', 'get', 'Caption'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip().split("\n")[1:]  # Skip the header line

        # Clean up the output and remove any empty lines
        deviceNames = [line.strip() for line in output if line.strip()]
        
            
            # Loop through possible OpenCV sources (device IDs 0-9)
        if len(deviceNames) == 0:
            cls.webcams.clear()
        for idx, name in enumerate(deviceNames):
                # Check if OpenCV can access the device
            cap = cv2.VideoCapture(idx)
            
            if cap.isOpened():
                cls.webcams.update({name : idx})
                cap.release()
            else:
                if name in list(cls.webcams.keys()):
                    del cls.webcams[name]
                print(cls.webcams)

def getWebcamDevicesThreadHandler():
    while True:
        webcamThreadHandler.getWebcamDevices()
        time.sleep(1)

WEBCAM_DETECTION_THREAD = threading.Thread(target=getWebcamDevicesThreadHandler, daemon=True)




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

"""model = YoloV1()
model.build((None, 448,448,3))
model.load_weights(DEFAULT_MODEL_PATH)"""
"""#load_model("E:\\IMPORTANT MODEL SAVES FOR NEA\\YOLOV1_v1.h5", custom_objects={"yoloLoss" : yoloLossPlaceholder, "yoloLoss" : yoloLossPlaceholder
                                                                                          ,"boundingBoxLoss" : yoloLossPlaceholder,
                                                                                          "ClassLoss" : yoloLossPlaceholder,
       
                                                                                          
                                                                                                                                                                                                                                                                "ConfidenceLoss" : yoloLossPlaceholder})"""

# class responsible for the GUI, inherits from tk.TK to allow tkinter to be utilised in OOP fashion
class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Object Detection")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.imageLabel = tk.Label(self)
        self.imageLabel.pack()
        self.model = YoloV1()
        self.model.build((None, 448,448,3))
        self.model.load_weights(DEFAULT_MODEL_PATH)
        self.detecting = False
        self.threshold = DEFAULT_THRESHOLD # if confidence > threshold then detection will display
        self.detectToggleButton = tk.Button(self, text="Toggle Detections", command=self.toggleDetection)
        self.detectToggleButton.pack()
        self.toggleLabel = tk.Label(self, text="Detection: Inactive", fg="red")
        self.toggleLabel.pack()
        self.bboxColour = DEFAULT_BBOX_COLOUR
        self.bboxWidth = DEFAULT_BBOX_WIDTH
        self.currentFrame = None
        self.frameGenerator = None
        self.setupMenu()
        self.maxFrameRate = 30
        self.updateInterval = 1000 // self.maxFrameRate  # Delay between frames in milliseconds
        self.classes = "person"
        self.webcamSource = DEFAULT_CAM_SOURCE
        self.webcamOptions = list(webcamThreadHandler.webcams.keys())
        self.webcamName = tk.StringVar()
        if len(self.webcamOptions) != 0: self.webcamName.set(self.webcamOptions[DEFAULT_CAM_SOURCE])
        self.currentSettingsWindow = None
        self.currentModelTrainer = None
        self.currentModelSettingsWindow = None



    # function used for toggle button's command, is called when the button is pressed
    def toggleDetection(self):
        if self.detecting is False:
            if self.model == None:
                messagebox.showerror(title="Model Error", message="No model is currently loaded. Please load a model.")
                return
            self.detecting = True
            self.toggleLabel.config(text="Detection: Active", fg="green")
        else:
            self.detecting = False
            self.toggleLabel.config(text="Detection: Inactive", fg="red")


        if self.currentFrame is not None:
            self.displayCurrentFrame()
        return


    # function used to declutter the constructor, sets all the properties of the different menus
    def setupMenu(self):
        self.menuBar = tk.Menu(self)
        self.fileMenu = tk.Menu(self.menuBar, tearoff=0)
        self.menuBar.add_cascade(label="Import", menu=self.fileMenu)
        self.fileMenu.add_cascade(label="Import Image", command=self.displayImage)
        self.fileMenu.add_cascade(label="Import Video", command=self.startDisplayVideo)
        self.fileMenu.add_cascade(label="Import Webcam Footage", command=self.startDisplayWebcamFootage)
        self.menuBar.add_cascade(label="Settings", command=self.openSettings)
        self.menuBar.add_cascade(label="Train New Model", command=self.openModelTrainer)
        self.config(menu=self.menuBar)


    def windowCheck(self, window):
        if window is not None and window.winfo_exists():
            messagebox.showerror(title="Window Already Open.", message="Window is already open. Please close before reopening.")
            return False
        return True

    def openSettings(self):
        if self.windowCheck(self.currentSettingsWindow):
            self.currentSettingsWindow = Settings(self)

    def openModelSettings(self):
        if self.windowCheck(self.currentModelSettingsWindow):
            self.currentModelSettingsWindow = ModelSettings(self)

    def openModelTrainer(self):
        if self.windowCheck(self.currentModelTrainer):
            self.currentModelTrainer = ModelTrainer(self)

    # function used for when importing a video
    def startDisplayVideo(self):
        self.frameGenerator = None
        videoPath = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video files", "*.MP4"),
                                                                                        ("Video files", "*.MKV"),
                                                                                          ("Video files", "*.MOV"),
                                                                                            ("Video files", "*.WEBM"),
                                                                                              ("All Files", "*.*")])
        self.frameGenerator = yieldNextFrame(videoDir=videoPath)
        self.displayFootage()


    # function used for starting webcam footage
    def startDisplayWebcamFootage(self):
        if len(webcamThreadHandler.webcams) == 0:
            messagebox.showerror(title="Webcam Device Error", message="No webcam devices have been detected, please try again.")
            return
        self.frameGenerator = None
        self.frameGenerator = yieldNextFrame(source=self.webcamSource)  # Initialize the generator for the webcam
        self.displayFootage()


    # function for all types of video footage including webcams, uses a generator to pass each frame
    # firstly through the model then draws the boxes onto the frame and displays it continuously
    def displayFootage(self):
        if self.frameGenerator is not None:
            try:
                # Get the next frame from the generator
                self.currentFrame = next(self.frameGenerator)
                
                if self.currentFrame:
                    self.displayCurrentFrame()
                else:
                    print("No frame received")
            except StopIteration:
                print("No more frames")
                self.frameGenerator = None  # Reset generator on completion
            except Exception as e:
                print(f"Error displaying frame: {e}")

            # Schedule the next frame update
            self.after(self.updateInterval, self.displayFootage)

    # loads an image in when user is importing image

    def displayImage(self):
        self.frameGenerator = None
        self.currentFrame = readImage(filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.JPG"), ("All Files", "*.*")]))
        width = self.currentFrame.width
        height = self.currentFrame.height
        while width > WINDOW_WIDTH or height > WINDOW_HEIGHT:
            self.currentFrame = self.currentFrame.resize((width // 2, height // 2))
            width = self.currentFrame.width
            height = self.currentFrame.height

        self.displayCurrentFrame()

        
        

    # a universal function for images AND videos, draws the model prediction onto the frame
    # then displays the current frame by updating the image label to contain the frame
    def displayCurrentFrame(self):
        frame = self.currentFrame.copy()
        if self.detecting:
            self.modelInputImage = (np.array(frame.resize((448,448))))[...,:3].reshape((1,448,448,3)).astype("float32") / 255.0
            frame = drawYoloBoxes(frame, self.model.predict(np.transpose(self.modelInputImage, (0, 2, 1, 3))), self, self.classes)
        self.photo = ImageTk.PhotoImage(frame)
        self.imageLabel.config(image=self.photo)
        self.imageLabel.pack()




# draws the bounding boxes based on the models prediction
def drawYoloBoxes(image, yoloPrediction, instance, classes="hawk tuah"):
    """
    Draw bounding boxes on the image based on YOLOv1 predictions.

    Args:
        image (PIL.Image): The original image.
        yoloPrediction (numpy.array): The YOLOv1 prediction with shape (1, s, s, 5*b + num_classes).
            Each grid cell contains `b` bounding boxes, each with [confidence, x_center, y_center, w, h],
            and class probabilities following the bounding boxes.
        instance (GUI object): Instance with model and settings for drawing bounding boxes.

    Returns:
        PIL.Image: The image with drawn bounding boxes.
    """
    # Get grid size and number of bounding boxes per grid cell
    s = instance.model.gridSize  # grid size (e.g., 7x7)
    b = instance.model.bboxes  # number of bounding boxes per grid cell (e.g., 2) 
    classes = classes.split(",")
    numOfClasses = len(classes)
    
    # Create a drawing context for the image
    draw = ImageDraw.Draw(image)
    
    # Image dimensions
    imgWidth, imgHeight = image.size

    # Grid cell size
    cellWidth = imgWidth / s
    cellHeight = imgHeight / s

    # YOLOv1 prediction output format: (1, s, s, 5*b + num_classes)
    output = yoloPrediction[0]

    # Loop through each grid cell
    for j in range(s):  # Iterate over the rows of grid cells
        for i in range(s):  # Iterate over the columns of grid cells
            for box in range(b):  # Iterate over each bounding box in the current grid cell
                # Extract confidence and bounding box details for this box
                confidence = output[i, j, box * 5]
                print(confidence)
                if confidence > instance.threshold:  # Draw only if confidence is above the threshold
                    xCentre = output[i, j, box * 5 + 1]
                    yCentre = output[i, j, box * 5 + 2]
                    box_w = output[i, j, box * 5 + 3]
                    box_h = output[i, j, box * 5 + 4]

                    # Convert normalized bounding box coordinates to image coordinates
                    w = cellWidth * box_w
                    h = cellHeight * box_h
                    x = (i * cellWidth) + (cellWidth * xCentre) - (w / 2)
                    y = (j * cellHeight) + (cellHeight * yCentre) - (h / 2)
                    x2 = x + w
                    y2 = y + h
                    classProbs = output[i, j, 5 * b:]  # Access class probabilities after all bounding boxes
                    classIndex = classProbs.argmax()  # Get the index of the class with the highest probability

                    # Draw the bounding box
                    draw.rectangle([x, y, x2, y2], outline=instance.bboxColour, width=instance.bboxWidth)

                    # Draw class label and confidence score (assuming a single class for simplicity)
                    draw.text([x2, y], text=f"{classes[classIndex]}: {confidence:.2f}", fill=instance.bboxColour)

    return image

class Settings(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.title("Settings")
        self.tempColour = self.master.bboxColour
        self.webcamOptions = list(webcamThreadHandler.webcams.keys())
        self.webcamName = tk.StringVar()
        if len(self.webcamOptions) != 0: self.webcamName.set(self.webcamOptions[DEFAULT_CAM_SOURCE])
        self.thresholdSlider = tk.Scale(self, from_=0, to=1.0, resolution=0.005, orient="horizontal", label="Confidence Threshold:", length=135)
        self.thresholdSlider.set(self.master.threshold)
        self.thresholdSlider.pack()
        self.frameRateSlider = tk.Scale(self, from_=1, to=60, resolution=1, orient="horizontal", label="Maximum Frame Rate:", length=135)
        self.frameRateSlider.set(self.master.maxFrameRate)
        self.frameRateSlider.pack()
        self.widthSlider = tk.Scale(self, from_=1, to=15, resolution=1, orient="horizontal", label="Bounding Box Width:", length=135)
        self.widthSlider.set(self.master.bboxWidth)
        self.widthSlider.pack()
        self.colourPicker = tk.Button(self, text="Change Bounding Box Colour", command=self.changeColour, fg=self.master.bboxColour)
        self.colourPicker.pack()
        self.webcamDropdownLabel = tk.Label(self, text="Webcam Device:")
        self.webcamDropdownLabel.pack()
        # self.webcamDropdown either a label or a dropdown depending on connected devices
        try:
            self.webcamDropdown = tk.OptionMenu(self, self.webcamName, *self.webcamOptions)
            self.webcamDropdown.pack()
            print("test")
        except Exception as exception:
            print(exception)
        if len(webcamThreadHandler.webcams) == 0:
            self.webcamDropdown = tk.Label(self, text="No webcam devices detected. Try reopening settings.", fg="red")
            self.webcamDropdown.pack()
        self.changeModelSettings = tk.Button(self, text="Model Settings", command=self.master.openModelSettings)
        self.changeModelSettings.pack()
        self.applySettingsButton = tk.Button(self, text="Apply Settings", command=self.applySettings)
        self.applySettingsButton.pack()


    def changeColour(self):
        self.tempColour = colorchooser.askcolor(title="Choose Colour")[1]
        self.colourPicker.config(fg=self.tempColour)
        

    # function is used for the button in settings, it applies all settings
    def applySettings(self):
        self.master.threshold = self.thresholdSlider.get()
        self.master.webcamSource = webcamThreadHandler.webcams.get(self.webcamName.get())
        self.master.maxFrameRate = self.frameRateSlider.get()
        self.master.updateInterval = 1000 // self.master.maxFrameRate
        self.master.bboxColour = self.tempColour
        self.master.bboxWidth = self.widthSlider.get()
        if self.master.currentFrame is not None:
            self.master.displayCurrentFrame()


class ModelSettings(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master.detecting = False
        self.master.toggleLabel.config(text="Detection: Inactive", fg="red")
        self.gridSizeInput = tk.Entry(self)
        self.boundingBoxesInput = tk.Entry(self)
        self.classCountInput = tk.Entry(self)
        self.classesInput = tk.Entry(self)
        tk.Label(self, text="Grid Size:").pack()
        self.gridSizeInput.pack()
        tk.Label(self, text="Bounding Boxes:").pack()
        self.boundingBoxesInput.pack()
        tk.Label(self, text="Number Of Classes:").pack()
        self.classCountInput.pack()
        tk.Label(self, text="Class Names:").pack()
        self.classesInput.pack()
        self.loadWeightsButton = tk.Button(self, text="Load Weights", command=self.loadWeights)
        self.modelLoadWarningLabel = tk.Label(self,
                                               text="Ensure you have entered the parameters BEFORE loading weights.",
                                               fg="red")
        self.loadWeightsButton.pack()
        self.modelLoadWarningLabel.pack()

    def loadWeights(self):
        self.master.model = None
        self.master.model = YoloV1(int(self.gridSizeInput.get()), int(self.classCountInput.get()), int(self.boundingBoxesInput.get()))
        self.master.model.build((None,448,448,3))
        self.master.classes = self.classesInput.get()
        weights = filedialog.askopenfile(filetypes=[("Model Weights File","*.h5"), ("All Files", "*.*")])
        print(weights)
        try:
            self.master.model.load_weights(weights.name.replace("\\","\\\\"))
        except ValueError:
            messagebox.showerror(title="Incompatible Weights",
                                      message="The weights do not correctly match the layers based on the parameters provided. Ensure parameters are correct and retry.")
            self.master.model = None
            return
    
        self.destroy()


class ModelTrainer(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.trainingDir = None
        self.outputDir = None
        self.trainingGridSizeInput = tk.Entry(self)
        self.trainingBoundingBoxesInput = tk.Entry(self)
        self.trainingClassCountInput = tk.Entry(self)
        tk.Label(self, text="Grid Size:").pack()
        self.trainingGridSizeInput.pack()
        tk.Label(self, text="Bounding Boxes:").pack()
        self.trainingBoundingBoxesInput.pack()
        tk.Label(self, text="Number Of Classes:").pack()
        self.trainingClassCountInput.pack()
        tk.Label(self, text="Input number of epochs:").pack()
        self.epochsInput = tk.Entry(self)
        self.epochsInput.pack()
        tk.Label(self, text="Input Learning rate:").pack()
        self.learningRateInput = tk.Entry(self)
        self.learningRateInput.pack()
        tk.Label(self, text="Input Batch Size:").pack()
        self.batchSizeInput = tk.Entry(self)
        self.batchSizeInput.pack()
        self.trainingDataInputButton = tk.Button(self, text="Input Training Folder", command=self.selectTrainingData)
        self.trainingDataInputButtonLabel = tk.Label(self, text="Currently Selected: None")
        self.trainingDataInputButton.pack()
        self.trainingDataInputButtonLabel.pack()
        self.outputButton = tk.Button(self, text="Input Output Folder", command=self.selectOutputFolder)
        self.outputButtonLabel = tk.Label(self, text="Currently Selected: None")
        self.outputButton.pack()
        self.outputButtonLabel.pack()
        self.beginTrainingButton = tk.Button(self, text="Begin Training", command=self.beginTraining)
        self.beginTrainingButton.pack()

    def selectTrainingData(self):
        try:
            self.trainingDir = filedialog.askdirectory(title="Select Training Folder:")
        except Exception:
            return
        self.trainingDataInputButtonLabel.config(text=f"Currently Selected: {self.trainingDir}")

    def selectOutputFolder(self):
        try:
            self.outputDir = filedialog.askdirectory(title="Select Training Annotations Folder:")
        except Exception:
            return
        self.outputButtonLabel.config(text=f"Currently Selected: {self.outputDir}")
    
        

    def beginTraining(self):
        self.currentInfoWindow = TrainingInfo(self, int(self.epochsInput.get()))
        model = YoloV1(int(self.trainingGridSizeInput.get()), int(self.trainingClassCountInput.get()), int(self.trainingBoundingBoxesInput.get()))
        self.trainer = ModelTraining(model,
                                     int(self.epochsInput.get()), int(self.batchSizeInput.get()), float(self.learningRateInput.get()), self.trainingDir,
                                     int(self.trainingGridSizeInput.get()), int(self.trainingBoundingBoxesInput.get()), int(self.trainingClassCountInput.get()), self.outputDir, self.currentInfoWindow)
        modelTrainingThread = threading.Thread(target=self.trainer.train) #
        modelTrainingThread.start()
class TrainingInfo(tk.Toplevel):
    def __init__(self, master, totalEpochs):
        super().__init__(master)
        self.epochLossContainer = [[], [], [], []]
        self.totalEpochs = totalEpochs #1 # default
        self.loss, self.confidenceLoss, self.classLoss, self.boundingBoxLoss = None, None,  None, None
        self.lossLabel = tk.Label(self, text=f"Current Loss: N/A")
        self.classLossLabel = tk.Label(self, text=f"Current Class Loss: N/A")
        self.confidenceLossLabel = tk.Label(self, text=f"Current Confidence Loss: N/A")
        self.boundingBoxLossLabel = tk.Label(self, text=f"Current Bounding Box Loss: N/A")
        self.epochProgressBar = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self.trainingProgressBar = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self.lossLabel.pack()
        self.classLossLabel.pack()
        self.confidenceLossLabel.pack()
        self.boundingBoxLossLabel.pack()
        tk.Label(self, text="Current Epoch Progress:").pack()
        self.epochProgressBar.pack()
        tk.Label(self, text="Current Training Progress:").pack()
        self.trainingProgressBar.pack()
        self.createPlot()
        self.lossGraphDisplay = FigureCanvasTkAgg(self.lossGraph, master=self)
        self.lossGraphWidget = self.lossGraphDisplay.get_tk_widget()
        self.lossGraphWidget.pack()
    
    def createPlot(self):
        self.lossGraph, self.lossGraphAxis = plt.subplots()
        self.lossGraphAxis.set_title("Epoch-Loss Graph")
        self.lossGraphAxis.set_xlabel("Epoch")
        self.lossGraphAxis.set_ylabel("Loss Value")
        self.lossGraphAxis.legend()

    def updatePlot(self):
        loss, confidenceLoss, classLoss, bboxLoss = self.epochLossContainer[0],self.epochLossContainer[1],self.epochLossContainer[2],self.epochLossContainer[3]
        self.lossGraphAxis.clear()
        #self.lossGraphAxis.plot(range(len(loss)), loss, color="red", label="YOLO Loss")
        self.lossGraphAxis.plot(range(len(classLoss)), classLoss, color="blue", label="Class Loss")
        self.lossGraphAxis.plot(range(len(confidenceLoss)), confidenceLoss, color="green", label="Confidence Loss")
        self.lossGraphAxis.plot(range(len(bboxLoss)), bboxLoss, color="orange", label="Bounding Box Loss")
        self.lossGraphAxis.legend()
        self.lossGraphDisplay.draw()

    



if __name__ == "__main__":
    webcamThreadHandler.getWebcamDevices()
    WEBCAM_DETECTION_THREAD.start()
    gui = GUI()
    gui.mainloop()
# THIS WORKS FIGURE OUT WHY!!!!!!!!
#TODO