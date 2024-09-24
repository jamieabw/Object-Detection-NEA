"""import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from videoHandler import yieldNextFrame, readImage

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Object Detection")
        self.geometry("800x600")
        self.imageLabel = tk.Label(self)
        self.genMenuBars()

    def genMenuBars(self):
        self.menuBar = tk.Menu(self)
        self.fileMenu = tk.Menu(self.menuBar, tearoff=0)
        self.menuBar.add_cascade(label="Import", menu=self.fileMenu)
        self.fileMenu.add_cascade(label="Import Image", command=self.displayImage)
        self.fileMenu.add_cascade(label="Import Video")
        self.fileMenu.add_cascade(label="Import Webcam Footage", command=self.startDisplayWebcamFootage)
        self.config(menu=self.menuBar)

    def displayImage(self):
        image = readImage(filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.JPG"), ("All Files", "*.*")]))
        width = image.width
        height = image.height
        image = image.resize((width // 8, height // 8))
        
        self.photo = ImageTk.PhotoImage(image)
        self.imageLabel.config(image=self.photo)
        self.imageLabel.pack()

    def startDisplayWebcamFootage(self):
        self.frameGenerator = yieldNextFrame()
        self.displayWebcamFootage()


    def displayWebcamFootage(self): # TODO: MAKE THIS CODE YOUR OWN
        if self.frameGenerator is not None:
            try:
                # Get the next frame from the generator
                frame = next(self.frameGenerator)
                
                if frame:
                    # Convert the frame to a format Tkinter can use
                    photo = ImageTk.PhotoImage(frame)
                    
                    # Update the label with the new frame
                    self.imageLabel.config(image=photo)
                    self.imageLabel.image = photo  # Keep a reference to avoid garbage collection
                else:
                    print("No frame received")
            except StopIteration:
                print("No more frames")
                self.frameGenerator = None  # Reset generator on completion
            except Exception as e:
                print(f"Error displaying frame: {e}")

            # Schedule the next frame update
            self.after(33, self.displayWebcamFootage)




# NOTE: this method will not work, it doesnt display each frame and instead displays nothing, it also reaches maximum recursion
# depth relatively fast so webcams cannot be on for a while




if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()"""

        
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import keras.regularizers
from videoHandler import yieldNextFrame, readImage
import tensorflow as tf
from keras.models import load_model
from keras import layers
import numpy as np
from CNNblocks import CNNBlock
import keras
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
DEFAULT_THRESHOLD = 0.4
CLASSES = 1
GRID_SIZE = 7
BBOXES = 1
KERNELS = [64, 192, 128, 256, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 512, 1024, 512, 1024, 512, 1024, 1024, 1024]
SIZES = [7, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3]
STRIDES = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
l2_regularizer = keras.regularizers.l2(0)

def yoloLossPlaceholder():
    pass

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

model = YoloV1()
model.build((None, 448,448,3))
model.load_weights("E:\\IMPORTANT MODEL SAVES FOR NEA\\YOLOV1_v4.h5")
"""#load_model("E:\\IMPORTANT MODEL SAVES FOR NEA\\YOLOV1_v1.h5", custom_objects={"yoloLoss" : yoloLossPlaceholder, "yoloLoss" : yoloLossPlaceholder
                                                                                          ,"boundingBoxLoss" : yoloLossPlaceholder,
                                                                                          "ClassLoss" : yoloLossPlaceholder,
                                                                                          "ConfidenceLoss" : yoloLossPlaceholder})"""

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
        self.currentFrame = None
        self.frameGenerator = None
        self.setupMenu()
        self.update_interval = 30  # Delay between frames in milliseconds

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

    def setupMenu(self):
        self.menuBar = tk.Menu(self)
        self.fileMenu = tk.Menu(self.menuBar, tearoff=0)
        self.menuBar.add_cascade(label="Import", menu=self.fileMenu)
        self.fileMenu.add_cascade(label="Import Image", command=self.displayImage)
        self.fileMenu.add_cascade(label="Import Video", command=self.startDisplayVideo)
        self.fileMenu.add_cascade(label="Import Webcam Footage", command=self.startDisplayWebcamFootage)
        self.menuBar.add_cascade(label="Settings", command=self.openSettings)
        self.config(menu=self.menuBar)

    def openSettings(self):
        self.settingsWindow = tk.Toplevel(self)
        self.settingsWindow.geometry("600x400")
        self.settingsWindow.title("Settings")
        self.thresholdSlider = tk.Scale(self.settingsWindow, from_=0, to=1.0, resolution=0.005, orient="horizontal", label="Confidence Threshold:", length=135)
        self.thresholdSlider.set(self.threshold)
        self.thresholdSlider.pack()
        self.applySettingsButton = tk.Button(self.settingsWindow, text="Apply Settings", command=self.applySettings)
        self.applySettingsButton.pack()

    def applySettings(self):
        self.threshold = self.thresholdSlider.get()
        if self.currentFrame is not None:
            self.displayCurrentFrame()



    def startDisplayVideo(self):
        self.frameGenerator = None
        videoPath = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video files", "*.MP4"), ("All Files", "*.*")])
        self.frameGenerator = yieldNextFrame(videoDir=videoPath)
        self.displayFootage()



    def startDisplayWebcamFootage(self):
        self.frameGenerator = None
        self.frameGenerator = yieldNextFrame(source=0)  # Initialize the generator for the webcam
        self.displayFootage()

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
            self.after(self.update_interval, self.displayFootage)

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

        
        


    def displayCurrentFrame(self):
        frame = self.currentFrame.copy()
        if self.detecting:
            self.modelInputImage = (np.array(frame.resize((448,448))))[...,:3].reshape((1,448,448,3)).astype("float32") / 255.0
            frame = draw_yolo_boxes(frame, model.predict(np.transpose(self.modelInputImage, (0, 2, 1, 3))), self)
        self.photo = ImageTk.PhotoImage(frame)
        self.imageLabel.config(image=self.photo)
        self.imageLabel.pack()


from PIL import Image, ImageDraw
import numpy as np

def draw_yolo_boxes(image, yolo_prediction, instance, s=7):
    """
    Draw bounding boxes on the image based on YOLOv1 predictions.
    
    Args:
        image (PIL.Image): The original image.
        yolo_prediction (numpy.array): The YOLOv1 prediction with shape (1, s, s, 6).
            Each grid cell contains [confidence, x_center, y_center, w, h, classes].
        s (int): The number of grid cells in one dimension (default is 7 for YOLOv1).
    
    Returns:
        PIL.Image: The image with drawn bounding boxes.
    """
    # Convert the image to an editable format
    draw = ImageDraw.Draw(image)
    
    # Image dimensions
    img_width, img_height = image.size
    
    # Grid cell size
    cell_width = img_width / s
    cell_height = img_height / s
    print(img_width, img_height)

    """

    TODO: CHANGE THIS CODE TO MINE AND ALSO MAKE IT WORK AS THE BOUNDING BOXES ARE NOT BEING GENERATED CORRECTLY
    """

    # Loop through each grid cell
    output = yolo_prediction[0]
    for j in range(s):
        for i in range(s):
            # Extract the bounding box data for the current grid cell
             # Ensure the shape is correct
                
            if output[i, j, 0] > instance.threshold:  # Draw only if confidence is above a threshold
                w = cell_width * output[i, j, 3]
                h = cell_height * output[i, j, 4]
                x = (i * cell_width) + (cell_width * output[i, j, 1]) - ((w)/2)
                y = (j * cell_height) + (cell_height * output[i, j, 2]) - ((h)/2)
                x2 = x + w
                y2 = y + h
                print("DETECTION!")
                    
                draw.rectangle([x, y, x2, y2], outline="red", width=2)

    return image


if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()
# THIS WORKS FIGURE OUT WHY!!!!!!!!
#TODO