import os
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
        return label
    

def convertToArray(imagePath, size=(448,448)):
    image = Image.open(imagePath)
    return np.transpose(np.array(image.resize(size)), (1,0,2)) # np.array turns the image into an array of [h,w,c] it needs to be [w,h,c]

def convertToImage(image):
    image *= 255
    image = np.array(image, dtype="uint8")
    return Image.fromarray(np.transpose(image, (1,0,2)))


def preprocessData(folderDir, testData=False):
    images =[] #np.array([])
    labels =[] #np.array([])
    counter = 0
    for file in os.listdir(folderDir):
        if "txt" in file:
            labels.append(encodeLabels(f"{folderDir}\\{file}", 8,1,1))
            #np.append(labels, encodeLabels(f"{folderDir}\\{file}", 8,1,1))
        else:
            if testData:
                convertToArray(images.append(f"{folderDir}\\{file}"))
            else:    
                images.append(f"{folderDir}\\{file}")
            #np.append(images, jpg_to_resized_array(f"{folderDir}\\{file}")
        counter +=1 
    images = np.array(images)
    labels = np.array(labels)

    print(images.shape)
    print(labels.shape)
    if testData:
        images = images.astype("float32") / 255.0
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