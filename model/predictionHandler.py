from dataProcessing import preprocessData, convertToImage
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from math import sqrt
cellSize = 448 / 7
testDir = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\dataset - CROWDHUMAN\\CrowdHuman_train01\\Images\\273271,1ba93000e00e011c.jpg"
classes = list(range(20))
classesNames = ["Person"]
"""classesNames = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]"""

def findBoxes(input, output):
    w,h,x,y = (0,0,0,0)
    bboxes = []
    output = output[0]
    image = convertToImage(input)
    for j in range(7):
        for i in range(7):
            if output[i, j, 0] <= 0.345 or output[i, j, 3] == 0 or output[i, j, 4] == 0:
                if output[i,j,0] != 0: print(output[i,j,0])
                continue

            w = cellSize * output[i, j, 3]
            h = cellSize * output[i, j, 4]
            x = (i * cellSize) + (cellSize * output[i, j, 1]) - ((w)/2)
            y = (j * cellSize) + (cellSize * output[i, j, 2]) - ((h)/2)
            class_probs = output[i, j, 5:]  # Assuming class probabilities start from index 6
            class_id = tf.argmax(class_probs)
            c = classes[class_id]


            bboxes.append([x,y,w,h,c])

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rects = []
    

    for bbox in bboxes:
        ax.add_patch(patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], edgecolor='red', facecolor="none"))
        label = f"{classesNames[int(bbox[4])]}"
        ax.text(bbox[0], bbox[1], label, fontsize=8, color='black')
        print((bbox[0],bbox[1]), bbox[2], bbox[3])
    #for rect in rects:
        #ax.add_patch(rect)

    plt.show()