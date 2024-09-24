import os
# for not crowdhuman the other one the format is: class, x, y, w, h but the bbox values are already normalised
import tensorflow as tf
import numpy as np
from random import shuffle
from PIL import Image
from math import floor
"""def encodeLabels(textFileDir, S, B, C):
    label = np.zeros(shape=[S,S,B * (C + 5)])
    cellSize = 1 / S
    with open(textFileDir, "r") as t:
        for line in t.readlines():
            properties = line.split(" ")
            if len(properties) < 4:
                break # THIS IS BROKEN, NEED TO FIND A WAY TO ENCODE LABELS WHERE THERE ARE NO OBJECTS
                #print(properties)
            cell = [int(float(properties[1]) // cellSize), int(float(properties[2]) // cellSize)]
            #print(cell)
            #print((float(properties[1])  % cellSize) / cellSize)
            label[cell[0], cell[1],0] = 1
            label[cell[0], cell[1],1] = (float(properties[1])  % cellSize) / cellSize
            label[cell[0], cell[1],2] = (float(properties[2])  % cellSize) / cellSize
            label[cell[0], cell[1],3] = float(properties[3]) / cellSize
            label[cell[0], cell[1],4] = float(properties[4]) / cellSize
            label[cell[0], cell[1],5] = 1
        return np.nan_to_num(label)"""
    


def encodeLabels(textFileDir, S, B, C):
    # Create label array: [S, S, B * (5 + C)]
    label = np.zeros(shape=[S, S, (B * 5) + C])
    cellSize = 1 / S
    
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
            for b in range(B):
                boxStart = b * (5 + C)  # Start index for this box in the label array
                
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
    

def convertToArray(imagePath, size=(448,448)):
    image = Image.open(imagePath)
    return np.transpose(np.array(image.resize(size)), (1,0,2))[:, :, :3] # np.array turns the image into an array of [h,w,c] it needs to be [w,h,c]

def convertToImage(image):
    image *= 255
    image = np.array(image, dtype="uint8")
    return Image.fromarray(np.transpose(image, (1,0,2)))


def preprocessData(folderDir, S, B, C, testData=False):
    images =[] #np.array([])
    labels =[] #np.array([])
    counter = 0
    for file in os.listdir(folderDir):
        if "txt" in file:
            labels.append(encodeLabels(f"{folderDir}\\{file}", S,B,C))
            #np.append(labels, encodeLabels(f"{folderDir}\\{file}", 8,1,1))
        else:
            counter +=1
            if testData:
                ...#convertToArray(images.append(f"{folderDir}\\{file}"))
                images.append(f"{folderDir}\\{file}")
            else:    
                images.append(f"{folderDir}\\{file}")
            #np.append(images, jpg_to_resized_array(f"{folderDir}\\{file}")

    
    images = np.array(images)
    labels = np.array(labels)
    print(labels.shape)
    # Generate a permutation of indices
    if not testData:
        indices = np.arange(len(images))
        np.random.shuffle(indices)

        # Apply the permutation to both arrays
        shuffled_images = images[indices]
        print(labels.shape)
        shuffled_labels = labels[indices, ...]
        return shuffled_images, shuffled_labels
    

# Shuffle images and labels using the permutation of indices
    return images, labels

"""def preprocessData(folderDir):
    images =[] #np.array([])
    labels =[] #np.array([])
    print("hrehehrererherer")
    print(folderDir)
    counter = 0
    for file in os.listdir(folderDir):
        if counter == 2000: break
        if "txt" in file:
            labels.append(encodeLabels(f"{folderDir}\\{file}", 8,1,1))
            #np.append(labels, encodeLabels(f"{folderDir}\\{file}", 8,1,1))
        else:
            images.append(convertToArray(f"{folderDir}\\{file}"))
            #np.append(images, jpg_to_resized_array(f"{folderDir}\\{file}")
        counter +=1 
    images = np.array(images)
    labels = np.array(labels)

    print(images.shape)
    print(labels.shape)
    images = images.astype("float32") / 255.0
    return images, labels"""
# size divided by cell size = bbox size relative to cells