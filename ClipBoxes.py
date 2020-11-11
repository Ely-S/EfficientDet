from tensorflow import keras
import tensorflow as tf

class ClipBoxes(keras.layers.Layer):
    """
    Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs

        print(image, boxes)
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())
        print('shap', shape)

        height = shape[1]
        width = shape[2]

        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width - 1)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height - 1)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width - 1)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height - 1)

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]
