import cv2
from PIL import Image
import numpy as np
# NOTE: laptop requires source=0 desktop source=1

# a generator function which gets the next frame and returns it after
# required preprocessing is done (otherwise colour channels will be mixed)
def yieldNextFrame(source=0, videoDir=None):
    if videoDir == None:
        capture = cv2.VideoCapture(source)
    else:
        capture = cv2.VideoCapture(videoDir)

    while True:
        ret, frame = capture.read()
        if not ret: # breaks when no frames left
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield convertToImage(frame)

    capture.release()
    return -1


def convertToImage(frame):
    return Image.fromarray(frame)

def readImage(imagePath):
    return Image.open(imagePath)
