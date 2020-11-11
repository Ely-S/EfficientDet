import tensorflow as tf

import math


class PriorProbability(tf.keras.initializers.Initializer):
    """
    This initializer is for the last conv layer before
    computing focal loss.
    """

    def __init__(self, probability=0.01):
        # this value comes from https://arxiv.org/pdf/1708.02002.pdf section 3.3
        self.probability = probability

    def __call__(self, shape, dtype=None):
        # See https://arxiv.org/pdf/1708.02002.pdf under Initialization
        # for an explanation.

        # result = np.ones(shape, dtype=np.float32
        #                  ) * -math.log((1 - self.probability) / self.probability)

        scalar = -math.log((1-self.probability) / self.probability)
        result = tf.constant(scalar, shape=shape,
                             name="PriorProbabilityInitializer")
        return result

    def get_config(self):
        return {
            'probability': self.probability
        }
