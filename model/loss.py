import tensorflow as tf

EPSILON =   1e-9
LAMBDA_NOOBJ = 0.5
LAMBDA_COORD = 5
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

    coordLoss = LAMBDA_COORD * tf.reduce_sum(objectMask * (tf.square(trueX - predictedX) + tf.square(trueY - predictedY)))
    sizeLoss = LAMBDA_COORD * tf.reduce_sum(objectMask * (tf.square(trueW - predictedW) + tf.square(trueH - predictedH)))
    
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



"""import tensorflow as tf

# Constants
LAMBDA_NOOBJ = 0.5
LAMBDA_COORD = 2
S = 8
B = 1
C = 1

def yoloLoss(yTrue, yPred):
    """
    #Compute YOLOv1 loss given true labels and predictions.
    
    #:param yTrue: Ground truth labels with shape [batch_size, S, S, B * (C + 5)]
    #:param yPred: Model predictions with shape [batch_size, S, S, B * (C + 5)]
    #:return: Total loss value
"""
    bbox_loss = boundingBoxLoss(yPred, yTrue)
    conf_loss = confidenceLoss(yPred, yTrue)
    class_loss = classLoss(yPred, yTrue)
    
    total_loss = bbox_loss + conf_loss + class_loss
    return total_loss

def boundingBoxLoss(yPred, yTrue):
    """
    #Compute bounding box loss component.
    
    #:param yPred: Model predictions with shape [batch_size, S, S, B * (C + 5)]
    #:param yTrue: Ground truth labels with shape [batch_size, S, S, B * (C + 5)]
    #:return: Bounding box loss value
"""
    true_confidence = yTrue[..., 0:1]
    object_mask = tf.cast(true_confidence > 0, tf.float32)
    
    predicted_boxes = tf.concat([yPred[..., 1:3], yPred[..., 3:5]], axis=-1)  # x, y, w, h
    true_boxes = tf.concat([yTrue[..., 1:3], yTrue[..., 3:5]], axis=-1)  # x, y, w, h
    
    # Coordinate loss
    coord_loss = LAMBDA_COORD * tf.reduce_sum(object_mask * tf.square(true_boxes - predicted_boxes))
    
    # Size loss (squared root of width and height)
    size_loss = LAMBDA_COORD * tf.reduce_sum(object_mask * (
        tf.square(tf.sqrt(true_boxes[..., 2:3]) - tf.sqrt(tf.abs(predicted_boxes[..., 2:3]))) +
        tf.square(tf.sqrt(true_boxes[..., 3:4]) - tf.sqrt(tf.abs(predicted_boxes[..., 3:4])))
    ))
    
    return coord_loss + size_loss

def confidenceLoss(yPred, yTrue):
    """
    #Compute confidence loss component.
    
    #:param yPred: Model predictions with shape [batch_size, S, S, B * (C + 5)]
    #:param yTrue: Ground truth labels with shape [batch_size, S, S, B * (C + 5)]
    #:return: Confidence loss value
"""
    predicted_confidence = yPred[..., 0:1]
    true_confidence = yTrue[..., 0:1]
    object_mask = tf.cast(true_confidence > 0, tf.float32)
    
    obj_conf_loss = tf.reduce_sum(object_mask * tf.square(true_confidence - predicted_confidence))
    no_obj_conf_loss = tf.reduce_sum((1 - object_mask) * LAMBDA_NOOBJ * tf.square(true_confidence - predicted_confidence))
    
    return obj_conf_loss + no_obj_conf_loss

def classLoss(yPred, yTrue):
    """
    #Compute class loss component.
    
    #:param yPred: Model predictions with shape [batch_size, S, S, B * (C + 5)]
    #:param yTrue: Ground truth labels with shape [batch_size, S, S, B * (C + 5)]
    #:return: Class loss value
"""
    true_confidence = yTrue[..., 0:1]
    object_mask = tf.cast(true_confidence > 0, tf.float32)
    
    predicted_classes = yPred[..., 5:]  # Assumes class predictions start after bounding box parameters
    true_classes = yTrue[..., 5:]  # Assumes class labels start after bounding box parameters
    
    return tf.reduce_sum(object_mask * tf.square(true_classes - predicted_classes))
"""