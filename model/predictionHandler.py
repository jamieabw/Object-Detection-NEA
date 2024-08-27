from dataProcessing import preprocessData, convertToImage
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
cellSize = 448 / 8
testDir = "C:\\Users\\jamie\\Documents\\CS NEA 24 25 source code\\datasets\\dataset - CROWDHUMAN\\CrowdHuman_train01\\Images\\273271,1ba93000e00e011c.jpg"

def findBoxes(input, output):
    w,h,x,y = (0,0,0,0)
    bboxes = []
    output = output[0]
    image = convertToImage(input)
    for j in range(8):
        for i in range(8):
            if output[i, j, 0] < 0.2 or output[i, j, 3] == 0 or output[i, j, 4] == 0:
                if output[i,j,0] != 0: print(output[i,j,0])
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