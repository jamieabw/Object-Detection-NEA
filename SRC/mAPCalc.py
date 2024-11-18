import numpy as np
import matplotlib.pyplot as plt
IoUThreshold = 0.1
predicted = np.random.rand(7, 7, 6)
true = predicted#np.random.rand(7, 7, 6)


"""
FIX THIS ITS VERY BROKEN, IM NOT SURE WHAT IS CAUSING IT PROBABLY TRY WITH ACTUAL PREDICTIONS AND LABELS TO SEE IF
ITS ME OR THIS SHITTY SUBROUTINE
"""

def calculatemAP(predictions, truth):
    TP = 0
    FP = 0
    FN = 0
    precision = []
    recall = []
    for j in range(7):
        for i in range(7):
            predictionCell = predictions[i][j]
            trueCell = truth[i][j]
            if predictionCell[0] < 0:
                continue
            IoU = calculateIoU(predictionCell, trueCell)
            if IoU >= IoUThreshold:
                if trueCell[0] == 1:
                    TP += 1
                
                else:
                    FP += 1
            else:
                if trueCell[0] == 1:
                    FN += 1
            if TP + FP == 0:
                precision.append(0)
            else:
                precision.append(TP / (TP + FP))
            if TP + FN == 0:
                recall.append(0)
            else:
                recall.append(TP / (TP + FN))
    print(precision)
    print(recall)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='o')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.show()



def calculateIoU(predBox, trueBox):
    conTrue, xTrue, yTrue, wTrue, hTrue, classTrue = trueBox
    conPred, xPred, yPred, wPred, hPred, classPred = predBox

    # Calculate corners of the bounding boxes
    topLeftTrue = [xTrue - (wTrue / 2), yTrue - (hTrue / 2)]
    bottomRightTrue = [xTrue + (wTrue / 2), yTrue + (hTrue / 2)]
    topLeftPred = [xPred - (wPred / 2), yPred - (hPred / 2)]
    bottomRightPred = [xPred + (wPred / 2), yPred + (hPred / 2)]

    # Calculate intersection rectangle
    xLeft = max(topLeftTrue[0], topLeftPred[0])
    yTop = max(topLeftTrue[1], topLeftPred[1])
    xRight = min(bottomRightTrue[0], bottomRightPred[0])
    yBottom = min(bottomRightTrue[1], bottomRightPred[1])

    # Check for no intersection
    if xRight <= xLeft or yBottom <= yTop:
        return 0.0

    # Calculate intersection area
    intersection = (xRight - xLeft) * (yBottom - yTop)

    # Calculate areas of the individual boxes
    areaTrue = wTrue * hTrue
    areaPred = wPred * hPred

    # Calculate union area
    union = areaTrue + areaPred - intersection

    # Calculate IoU
    iou = intersection / union
    return iou



print(predicted)
calculatemAP(predicted, true)