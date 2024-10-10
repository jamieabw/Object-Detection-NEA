
import tkinter as tk
from tkinter import filedialog
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



"""
TODO: potential additions for settings: a colour picker to change the colour of bounding boxes, opacity too maybe
an outline size for bounding boxes
option to display confidence of the bounding box
advanced options to import a model and change hyperparameters of YOLOv1 (S, B, C values)
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
        #cls.webcams.clear()
        result = subprocess.run(['wmic', 'path', 'Win32_PnPEntity', 'where', 'Caption like "%cam%"', 'get', 'Caption'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip().split("\n")[1:]  # Skip the header line

            # Clean up the output and remove any empty lines
        deviceNames = [line.strip() for line in output if line.strip()]
        
        print("Available webcams:")
            
            # Loop through possible OpenCV sources (device IDs 0-9)
        for idx, name in enumerate(deviceNames):
            print(f"Webcam source ID: {idx}, Name: {name}")
                # Check if OpenCV can access the device
            cap = cv2.VideoCapture(idx)
            
            if cap.isOpened():
                print(f"OpenCV can access webcam {idx}: {name}")
                cls.webcams.update({name : idx})
                cap.release()
            else:
                print(f"OpenCV cannot access webcam {idx}: {name}")
                if name in list(cls.webcams.keys()):
                    del cls.webcams[name]
                #webcams.update({name : (idx, 0)})

        print(cls.webcams)


def getWebcamDevicesThreadHandler():
    while True:
        webcamThreadHandler.getWebcamDevices()
        time.sleep(1)

WEBCAM_DETECTION_THREAD = threading.Thread(target=getWebcamDevicesThreadHandler, daemon=True)




class YoloV1(tf.keras.Model):
    def __init__(self, grid_size=GRID_SIZE, classes=CLASSES, bboxes=BBOXES):
        super(YoloV1, self).__init__()
        self.grid_size = grid_size
        
        self.classes = classes
        self.bboxes = bboxes
        self.convLayers = CNNBlock(KERNELS, SIZES, STRIDES, l2Regularizer)
        self.denseLayer = layers.Dense(4096, kernel_regularizer=l2Regularizer)
        self.outputDense = layers.Dense(self.grid_size * self.grid_size * ((5 * self.bboxes) + self.classes), kernel_regularizer=l2Regularizer) # adding a relu activation to this breaks the loss i think - dunno why

    def call(self, inputs):
        x = self.convLayers(inputs)
        x = layers.Flatten()(x)
        x = self.denseLayer(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        #x = layers.Dropout(0.5)(x)
        x = self.outputDense(x)
        x =  layers.Reshape((self.grid_size, self.grid_size, (5 * self.bboxes) + self.classes))(x)
        return x

model = YoloV1()
model.build((None, 448,448,3))
model.load_weights("C:\\Users\\jamie\\Desktop\\saVES\\YOLOV1_v5.h5")
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
        self.detecting = False
        self.threshold = DEFAULT_THRESHOLD # if confidence > threshold then detection will display
        self.detectToggleButton = tk.Button(self, text="Toggle Detections", command=self.toggleDetection)
        self.detectToggleButton.pack()
        self.toggleLabel = tk.Label(self, text="Detection: Inactive", fg="red")
        self.toggleLabel.pack()
        self.bboxColour = DEFAULT_BBOX_COLOUR
        self.currentFrame = None
        self.frameGenerator = None
        self.setupMenu()
        self.maxFrameRate = 30
        self.updateInterval = 1000 // self.maxFrameRate  # Delay between frames in milliseconds
        self.webcamSource = DEFAULT_CAM_SOURCE
        self.webcamOptions = list(webcamThreadHandler.webcams.keys())
        self.webcamName = tk.StringVar()
        if len(self.webcamOptions) != 0: self.webcamName.set(self.webcamOptions[DEFAULT_CAM_SOURCE])


    # function used for toggle button's command, is called when the button is pressed
    def toggleDetection(self):
        if self.detecting is False:
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
        self.config(menu=self.menuBar)


    # function used in the settings tab, opens a toplevel where you can control the confidence threshold (add more here asp)
    def openSettings(self):
        #self.webcamDropdown = None
        self.webcamOptions = list(webcamThreadHandler.webcams.keys())
        self.webcamName = tk.StringVar()
        if len(self.webcamOptions) != 0: self.webcamName.set(self.webcamOptions[DEFAULT_CAM_SOURCE])
        self.settingsWindow = tk.Toplevel(self)
        self.settingsWindow.geometry("600x400")
        self.settingsWindow.title("Settings")
        self.thresholdSlider = tk.Scale(self.settingsWindow, from_=0, to=1.0, resolution=0.005, orient="horizontal", label="Confidence Threshold:", length=135)
        self.thresholdSlider.set(self.threshold)
        self.thresholdSlider.pack()
        self.frameRateSlider = tk.Scale(self.settingsWindow, from_=1, to=60, resolution=1, orient="horizontal", label="Maximum Frame Rate:", length=135)
        self.frameRateSlider.set(self.maxFrameRate)
        self.frameRateSlider.pack()
        self.webcamDropdownLabel = tk.Label(self.settingsWindow, text="Webcam Device:")
        self.webcamDropdownLabel.pack()
        #print(len(webcamThreadHandler.webcams))
        try:
            self.webcamDropdown = tk.OptionMenu(self.settingsWindow, self.webcamName, *self.webcamOptions)
            self.webcamDropdown.pack()
        except Exception as a:
            ...
        if len(webcamThreadHandler.webcams) == 0:
            self.webcamDropdown = tk.Label(self.settingsWindow, text="No webcam devices detected. Try reopening settings.", fg="red")
            self.webcamDropdown.pack()
        self.applySettingsButton = tk.Button(self.settingsWindow, text="Apply Settings", command=self.applySettings)
        self.applySettingsButton.pack()

        


    # function is used for the button in settings, it applies all settings
    def applySettings(self):
        self.threshold = self.thresholdSlider.get()
        self.webcamSource = webcamThreadHandler.webcams.get(self.webcamName.get())
        self.maxFrameRate = self.frameRateSlider.get()
        self.updateInterval = 1000 // self.maxFrameRate
        if self.currentFrame is not None:
            self.displayCurrentFrame()


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
            frame = drawYoloBoxes(frame, model.predict(np.transpose(self.modelInputImage, (0, 2, 1, 3))), self)
        self.photo = ImageTk.PhotoImage(frame)
        self.imageLabel.config(image=self.photo)
        self.imageLabel.pack()


from PIL import Image, ImageDraw
import numpy as np


# draws the bounding boxes based on the models prediction
def drawYoloBoxes(image, yoloPrediction, instance, s=7):
    """
    Draw bounding boxes on the image based on YOLOv1 predictions.
    
    Args:
        image (PIL.Image): The original image.
        yolo_prediction (numpy.array): The YOLOv1 prediction with shape (1, s, s, 6).
            Each grid cell contains [confidence, x_center, y_center, w, h, classes].
        instance (GUI object): will draw the box on the correct window
        s (int): The number of grid cells in one dimension (default is 7 for YOLOv1).
    
    Returns:
        PIL.Image: The image with drawn bounding boxes.
    """
    # Convert the image to an editable format
    draw = ImageDraw.Draw(image)
    
    # Image dimensions
    imgWidth, imgHeight = image.size
    
    # Grid cell size
    cellWidth = imgWidth / s
    cellHeight = imgHeight / s
    print(imgWidth, imgHeight)

    """

    TODO: CHANGE THIS CODE TO MINE AND ALSO MAKE IT WORK AS THE BOUNDING BOXES ARE NOT BEING GENERATED CORRECTLY
    """

    # Loop through each grid cell
    output = yoloPrediction[0]
    for j in range(s):
        for i in range(s):
            # Extract the bounding box data for the current grid cell
             # Ensure the shape is correct
                
            if output[i, j, 0] > instance.threshold:  # Draw only if confidence is above a threshold
                w = cellWidth * output[i, j, 3]
                h = cellHeight * output[i, j, 4]
                x = (i * cellWidth) + (cellWidth * output[i, j, 1]) - ((w)/2)
                y = (j * cellHeight) + (cellHeight * output[i, j, 2]) - ((h)/2)
                x2 = x + w
                y2 = y + h
                # print("DETECTION!") debugging statement
                    
                draw.rectangle([x, y, x2, y2], outline=instance.bboxColour, width=2)

    return image


if __name__ == "__main__":
    webcamThreadHandler.getWebcamDevices()
    WEBCAM_DETECTION_THREAD.start()
    gui = GUI()
    gui.mainloop()
# THIS WORKS FIGURE OUT WHY!!!!!!!!
#TODO