import tensorflow as tf
import numpy as np

# essential yolo loss combines 3 types of different losses and weights them accordingly to importance
# usually the bbox loss will have the highest loss and class loss will be the smallest loss


class YoloLoss:
    epsilon = 1e-2
    def __init__(self, B=1):
        self.B = B

    def __call__(self, yTrue, yPred):
        confidenceLoss = self.ConfidenceLoss(yTrue, yPred)
        coordLoss = self.boundingBoxLoss(yTrue, yPred)
        classLoss = self.ClassLoss(yTrue, yPred)
        totalLoss = (3 * coordLoss) + (1 * confidenceLoss) + (1 * classLoss)
        return totalLoss
    
    # uses mean squared error loss 
    def ConfidenceLoss(self, yTrue, yPred):
        existsObject = tf.expand_dims(yTrue[..., 0], -1)
        confidenceLoss = 0
        for b in range(self.B):
            confidenceLoss += tf.reduce_sum(tf.square(existsObject * (yTrue[..., b * 5:(b * 5) + 1] - yPred[..., b * 5:(b * 5) + 1])))
            confidenceLoss += 0.5 * tf.reduce_sum(tf.square((1 - existsObject) * (yTrue[..., b * 5:(b * 5) + 1] - yPred[..., b * 5:(b * 5) + 1])))
        tf.debugging.assert_all_finite(confidenceLoss, 'NaNs or Infs found in c')
        
        # add epsilon to avoid division by zero
        nonZeroCount = tf.cast(tf.math.count_nonzero(existsObject), dtype=tf.float32) + YoloLoss.epsilon
        return confidenceLoss / (nonZeroCount)

    def ClassLoss(self, yTrue, yPred):
        existsObject = tf.expand_dims(yTrue[..., 0], -1)
        classLoss = tf.reduce_sum(tf.square(existsObject * (yTrue[..., 5 * self.B:] - yPred[..., 5 * self.B:])))
        nonZeroCount = tf.cast(tf.math.count_nonzero(existsObject), dtype=tf.float32) + YoloLoss.epsilon
        return classLoss / (nonZeroCount)

    def boundingBoxLoss(self, yTrue, yPred):
        existsObject = tf.expand_dims(yTrue[..., 0], -1)
        xyLoss = 0
        whLoss = 0
        for b in range(self.B):
            xyPred = existsObject * yPred[..., (b * 5) + 1:(b * 5) + 3]
            xyTrue = existsObject * yTrue[..., (b * 5) + 1:(b * 5) + 3]
            tf.debugging.assert_all_finite(xyTrue, 'NaNs or Infs found in XYT')
            # allowing the square root of negative numbers to result in a negative by using signs   
            whPred = existsObject * tf.math.sign(yPred[..., (b * 5) + 3:(b * 5) + 5]) * tf.sqrt(tf.math.abs(yPred[..., (b * 5) + 3:(b * 5) + 5]))
            whTrue = existsObject * tf.sqrt(yTrue[..., (b * 5) + 3:(b * 5) + 5])
            tf.debugging.assert_all_finite(whTrue, 'NaNs or Infs found in whT')
            
            xyLoss += tf.reduce_sum(tf.square(xyPred - xyTrue))
            whLoss += tf.reduce_sum(tf.square(whPred - whTrue))
        tf.debugging.assert_all_finite(whLoss, 'NaNs or Infs found in wh')
        tf.debugging.assert_all_finite(xyLoss, f'NaNs or Infs found in xy {xyLoss}')
        tf.debugging.assert_all_finite(xyLoss, 'NaNs or Infs found in xy')        
        nonZeroCount = tf.cast(tf.math.count_nonzero(existsObject), dtype=tf.float32) + YoloLoss.epsilon
        return (xyLoss + whLoss) / (nonZeroCount)