# import keras
from tensorflow import keras
import tensorflow as tf

# for compat with refactor
from RegressBoxes import RegressBoxes
from ClipBoxes import ClipBoxes
from FilterDetections import FilterDetections


class BatchNormalization(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds
    the option to freeze parameters.
    """

    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, inputs, training=None, **kwargs):
        # return super.call, but set training
        if training:
            return super(BatchNormalization, self).call(
                inputs, training=(not self.freeze)
            )
        else:
            return super(BatchNormalization, self).call(inputs, training=False)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        # @NOTE: TFJS Converter does not understand this
        config.update({"freeze": self.freeze})
        return config


class wBiFPNAdd(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(
            name=self.name,
            shape=(num_in,),
            initializer=keras.initializers.constant(1 / num_in),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs, **kwargs):
        w = keras.activations.relu(self.w)
        elementwise_multiply = [w[i] * inputs[i] for i in range(len(inputs))]
        x = tf.reduce_sum(elementwise_multiply, axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

