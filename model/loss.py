import tensorflow as tf


LAMBDA_NOOBJ = 0.5
LAMBDA_COORD = 2
S = 8
B = 1
C = 1
"""
#each grid cell tensor contains:
#C, x1, y1, w1, h1, P(c1)
"""


def yoloLoss(yTrue, yPred):
    bboxLoss = boundingBoxLoss(yPred, yTrue)
    confLoss = confidenceLoss(yPred, yTrue)
    clsLoss = classLoss(yPred, yTrue)
    
    totalLoss = bboxLoss + confLoss + clsLoss
    return totalLoss

def boundingBoxLoss(yPred, yTrue):
    trueConfidence = yTrue[..., 0:1]
    objectMask = tf.cast(trueConfidence > 0, tf.float32)
    predictedX = yPred[..., 1:2]
    predictedY = yPred[..., 2:3]
    predictedW = yPred[..., 3:4]
    predictedH = yPred[..., 4:5]

    trueX = yTrue[..., 1:2]
    trueY = yTrue[..., 2:3]
    trueW = yTrue[..., 3:4]
    trueH = yTrue[..., 4:5]
    # soon for both of the losses here you can multiply them by the true confidence, for each grid cell (either 0 or 1)
 
    coordLoss = LAMBDA_COORD * tf.reduce_sum(objectMask * tf.square(trueX - predictedX) + tf.square(trueY - predictedY))
    sizeLoss = LAMBDA_COORD * tf.reduce_sum(objectMask * tf.square(tf.sqrt(trueW) - tf.sqrt(tf.abs(predictedW))) + tf.square(tf.sqrt(trueH) - tf.sqrt(tf.abs(predictedH))))
    return coordLoss + sizeLoss

def confidenceLoss(yPred, yTrue):
    predictedConfidence = yPred[..., 0:1]
    trueConfidence = yTrue[..., 0:1]
    objectMask = tf.cast(trueConfidence > 0, tf.float32)
    objConfidenceLoss = tf.reduce_sum(objectMask * tf.square(trueConfidence - predictedConfidence))
    noObjConfidenceLoss = tf.reduce_sum((1-objectMask) * LAMBDA_NOOBJ * tf.square(trueConfidence - predictedConfidence))
    return objConfidenceLoss + noObjConfidenceLoss

def classLoss(yPred, yTrue):
    trueConfidence = yTrue[..., 0:1]
    objectMask = tf.cast(trueConfidence > 0, tf.float32)
    predictedClasses = yPred[..., 5:]
    trueClasses = yTrue[..., 5:]
    return tf.reduce_sum(objectMask * tf.square(trueClasses - predictedClasses))


