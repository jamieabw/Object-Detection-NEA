import tensorflow as tf
def yoloLoss(yTrue, yPred):
    confidenceLoss = ConfidenceLoss(yTrue, yPred)
    coordLoss = boundingBoxLoss(yTrue, yPred)
    classLoss = ClassLoss(yTrue, yPred)
    totalLoss = (3 * coordLoss) + (1 * confidenceLoss) + (1 * classLoss)
    return totalLoss

def ConfidenceLoss(yTrue, yPred):
    existsObject = tf.expand_dims(yTrue[..., 0], -1)
    confidenceLoss = tf.reduce_sum(tf.square(existsObject * (yTrue[..., 0:1] - yPred[..., 0:1])))
    confidenceLoss += 0.5 * tf.reduce_sum(tf.square((1 - existsObject) * (yTrue[..., 0:1] - yPred[..., 0:1])))
    tf.debugging.assert_all_finite(confidenceLoss, 'NaNs or Infs found in c')
    
    # Add epsilon to avoid division by zero
    non_zero_count = tf.cast(tf.math.count_nonzero(existsObject), dtype=tf.float32)
    return confidenceLoss / (non_zero_count)

def ClassLoss(yTrue, yPred):
    existsObject = tf.expand_dims(yTrue[..., 0], -1)
    classLoss = tf.reduce_sum(tf.square(existsObject * (yTrue[..., 5:] - yPred[..., 5:])))
    
    # Add epsilon to avoid division by zero
    non_zero_count = tf.cast(tf.math.count_nonzero(existsObject), dtype=tf.float32)
    return classLoss / (non_zero_count)

def boundingBoxLoss(yTrue, yPred):
    existsObject = tf.expand_dims(yTrue[..., 0], -1)
    
    xyPred = existsObject * yPred[..., 1:3]
    xyTrue = existsObject * yTrue[..., 1:3]
    tf.debugging.assert_all_finite(xyTrue, 'NaNs or Infs found in XYT')
    
    # Ensure non-negative width and height before square root
    whPred = existsObject * tf.math.sign(yPred[..., 3:5]) * tf.sqrt(tf.math.abs(yPred[..., 3:5]))
    #whPred = existsObject * tf.sqrt(tf.maximum(yPred[..., 3:5], 0.0))
    whTrue = existsObject * tf.sqrt(yTrue[..., 3:5])
    tf.debugging.assert_all_finite(whTrue, 'NaNs or Infs found in whT')
    
    xyLoss = tf.reduce_sum(tf.square(xyPred - xyTrue))
    whLoss = tf.reduce_sum(tf.square(whPred - whTrue))
    tf.debugging.assert_all_finite(whLoss, 'NaNs or Infs found in wh')
    tf.debugging.assert_all_finite(xyLoss, f'NaNs or Infs found in xy {xyLoss}')
    tf.debugging.assert_all_finite(xyLoss, 'NaNs or Infs found in xy')
    
    # Add epsilon to avoid division by zero
    non_zero_count = tf.cast(tf.math.count_nonzero(existsObject), dtype=tf.float32)
    return 2 * (xyLoss + whLoss) / (non_zero_count)
"""EPSILON =   1e-9
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

def boundingBoxLoss(yTrue, yPred):
    existsObject = tf.expand_dims(yTrue[..., 0], -1)
    xyPred = existsObject * yPred[..., 1:3]
    xyTrue = existsObject * yTrue[..., 1:3]
    whPred = existsObject * tf.math.sign(yPred[..., 3:5]) * tf.sqrt(tf.math.abs(yPred[..., 3:5]))
    whTrue = existsObject * yTrue[..., 3:5]
    return 2 * (tf.reduce_sum(tf.math.square(whPred - whTrue)) + tf.reduce_sum(tf.math.square(xyPred - xyTrue))) / (tf.cast(tf.math.count_nonzero(existsObject), dtype=tf.float32))


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
    return tf.reduce_sum(objectMask * tf.square(trueClasses - predictedClasses))"""
"""



import tensorflow as tf

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