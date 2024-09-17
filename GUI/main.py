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
from videoHandler import yieldNextFrame, readImage
import tensorflow as tf
from keras.models import load_model
import keras
import numpy as np
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800

def yoloLossPlaceholder():
    pass

model = load_model("E:\\IMPORTANT MODEL SAVES FOR NEA\\YOLOV1_v1.h5", custom_objects={"yoloLoss" : yoloLossPlaceholder, "yoloLoss" : yoloLossPlaceholder
                                                                                          ,"boundingBoxLoss" : yoloLossPlaceholder,
                                                                                          "ClassLoss" : yoloLossPlaceholder,
                                                                                          "ConfidenceLoss" : yoloLossPlaceholder})

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Object Detection")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.imageLabel = tk.Label(self)
        self.imageLabel.pack()
        self.detecting = False
        self.detectToggleButton = tk.Button(self, text="Toggle Detections", command=self.toggleDetection)
        self.detectToggleButton.pack()
        self.frameGenerator = None
        self.setupMenu()
        self.update_interval = 30  # Delay between frames in milliseconds

    def toggleDetection(self):
        print(self.detecting)
        if self.detecting is False:
            self.detecting = True
        else:
            self.detecting = False

        return

    def setupMenu(self):
        self.menuBar = tk.Menu(self)
        self.fileMenu = tk.Menu(self.menuBar, tearoff=0)
        self.menuBar.add_cascade(label="Import", menu=self.fileMenu)
        self.fileMenu.add_cascade(label="Import Image", command=self.displayImage)
        self.fileMenu.add_cascade(label="Import Video", command=self.startDisplayVideo)
        self.fileMenu.add_cascade(label="Import Webcam Footage", command=self.startDisplayWebcamFootage)
        self.config(menu=self.menuBar)


    def startDisplayVideo(self):
        self.frameGenerator = None
        videoPath = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video files", "*.MP4"), ("All Files", "*.*")])
        self.frameGenerator = yieldNextFrame(videoDir=videoPath)
        self.displayFootage()



    def startDisplayWebcamFootage(self):
        self.frameGenerator = None
        self.frameGenerator = yieldNextFrame(source=1)  # Initialize the generator for the webcam
        self.displayFootage()

    def displayFootage(self):
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
            self.after(self.update_interval, self.displayFootage)

    def displayImage(self):
        self.frameGenerator = None
        image = readImage(filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.JPG"), ("All Files", "*.*")]))
        width = image.width
        height = image.height
        while width > WINDOW_WIDTH or height > WINDOW_HEIGHT:
            image = image.resize((width // 2, height // 2))
            width = image.width
            height = image.height

        if self.detecting:
            himage = (np.array(image.resize((448,448))) / 255.0)[...,:3].reshape((1,448,448,3))
            image = draw_yolo_boxes(image, model.predict(himage))

        
        
        self.photo = ImageTk.PhotoImage(image)
        self.imageLabel.config(image=self.photo)
        self.imageLabel.pack()


from PIL import Image, ImageDraw
import numpy as np

def draw_yolo_boxes(image, yolo_prediction, s=7):
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

    """

    TODO: CHANGE THIS CODE TO MINE AND ALSO MAKE IT WORK AS THE BOUNDING BOXES ARE NOT BEING GENERATED CORRECTLY
    """

    # Loop through each grid cell
    for j in range(s):
        for i in range(s):
            # Extract the bounding box data for the current grid cell
            cell_data = yolo_prediction[0, j, i]
            
            if cell_data.shape[0] == 6:  # Ensure the shape is correct
                confidence, x_center, y_center, box_width, box_height, classes = cell_data
                
                if confidence > 0.4:  # Draw only if confidence is above a threshold
                    w = box_width * cell_width
                    h = box_height * cell_height
                    x1 = (i * cell_width) + (x_center * cell_width) - (w/2)
                    y1 = (j * cell_width) + (x_center * cell_width) - (h/2)
                    x2 = x1 + w
                    y2 = y1 + h
                    
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    return image


if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()
# THIS WORKS FIGURE OUT WHY!!!!!!!!
#TODO