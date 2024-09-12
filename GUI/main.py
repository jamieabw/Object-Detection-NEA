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
from PIL import Image, ImageTk
from videoHandler import yieldNextFrame, readImage
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Object Detection")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.imageLabel = tk.Label(self)
        self.imageLabel.pack()
        self.frameGenerator = None
        self.setupMenu()
        self.update_interval = 30  # Delay between frames in milliseconds

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
        self.frameGenerator = yieldNextFrame(source=0)  # Initialize the generator for the webcam
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
        
        
        self.photo = ImageTk.PhotoImage(image)
        self.imageLabel.config(image=self.photo)
        self.imageLabel.pack()

if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()
# THIS WORKS FIGURE OUT WHY!!!!!!!!
#TODO