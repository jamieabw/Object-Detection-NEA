from dataProcessing import preprocessData
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
cellSize = 448 / 8

testDir = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\dataset-humans\\INRIA Person detection dataset.v1i.darknet\\test\\crop001712_jpg.rf.5035be3fa00b6ba52e43b8ba0e2fb762.jpg"
def findBoxes(output):
    bboxes = []
    output = output[0]
    image = Image.open(testDir).resize((448,448))
    for j in range(8):
        for i in range(8):
            if output[i, j, 0] < 0.2 or output[i, j, 3] == 0 or output[i, j, 4] == 0:
                continue
            w = cellSize * output[i, j, 3]
            h = cellSize * output[i, j, 4]
            x = (i * cellSize) + (cellSize * output[i, j, 1]) - (w/2)
            y = (j * cellSize) + (cellSize * output[i, j, 2]) - (h/2)
            bboxes.append([x,y,w,h])
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rects = []
    for bbox in bboxes:
        ax.add_patch(patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], edgecolor='red', facecolor="none"))
        print((bbox[0],bbox[1]), bbox[2], bbox[3])
    #for rect in rects:
        #ax.add_patch(rect)

    plt.show()