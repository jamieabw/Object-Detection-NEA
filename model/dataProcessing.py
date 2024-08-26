# for not crowdhuman the other one the format is: class, x, y, w, h but the bbox values are already normalised
import tensorflow as tf
import numpy as np
from PIL import Image
def encodeLabels(textFileDir, S, B, C):
    label = np.zeros(shape=[S,S,B * (C + 5)])
    cellSize = 1 / S
    with open(textFileDir, "r") as t:
        for line in t.readlines():
            properties = line.split(" ")
            print(properties)
            cell = [int(float(properties[1]) // cellSize), int(float(properties[2]) // cellSize)]
            print(cell)
            print((float(properties[1])  % cellSize) / cellSize)
            label[cell[0], cell[1],0] = 1
            label[cell[0], cell[1],1] = (float(properties[1])  % cellSize) / cellSize
            label[cell[0], cell[1],2] = (float(properties[2])  % cellSize) / cellSize
            label[cell[0], cell[1],3] = float(properties[3]) / cellSize
            label[cell[0], cell[1],4] = float(properties[4]) / cellSize
            label[cell[0], cell[1],5] = 1
        return label
    

def jpg_to_resized_array(image_path, size=(448, 448)):
    """
    Resize a JPG image to the specified size and convert it to a NumPy array with dimensions (width, height, channels).

    Args:
        image_path (str): The path to the JPG image file.
        size (tuple): The target size as a tuple (width, height). Default is (448, 448).

    Returns:
        np.array: The resized image as a NumPy array with shape (width, height, channels).
    """
    # Open the image file
    img = Image.open(image_path)
    
    # Resize the image
    img_resized = img.resize(size)
    
    # Convert the resized image to a NumPy array
    img_array = np.array(img_resized)
    
    # Reorder dimensions from (height, width, channels) to (width, height, channels)
    img_array = np.transpose(img_array, (1, 0, 2))
    
    return img_array
            


dir = "C:\\Users\\jamie\Documents\\CS NEA 24 25 source code\\src\\model\\testdata\\crop_000003_jpg.rf.df647b1de3032ae2eff2f38159b2be1c.txt"
imgDir = "C:\\Users\\jamie\Documents\\CS NEA 24 25 source code\\src\\model\\testdata\\crop_000003_jpg.rf.df647b1de3032ae2eff2f38159b2be1c.jpg"
a = encodeLabels(dir, 8, 1, 1)
print(jpg_to_resized_array(imgDir).shape)
print(a.shape)
print(a[4,3])
# size divided by cell size = bbox size relative to cells