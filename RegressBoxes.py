from tensorflow import keras
import numpy as np

# These are the default arguments for regress boxes.
# They are immutable because default arguments should never
# be mutated
default_mean = np.array([0, 0, 0, 0], dtype="float32")
default_std = np.array([0.2, 0.2, 0.2, 0.2], dtype="float32")
default_mean.setflags(write=False)
default_std.setflags(write=False)


class RegressBoxes(keras.layers.Layer):
    """
    Keras layer for applying regression values to boxes.
    """

    dtype = "float32"
    saved_anchors = False

    def __init__(
        self, mean=default_mean, std=default_std, anchor_shape=None, *args, **kwargs
    ):
        """
        Initializer for the RegressBoxes layer.

        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError(
                "Expected mean to be a np.ndarray,"
                f" list or tuple. Received: {type(mean)}"
            )

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError(
                "Expected std to be a np.ndarray,"
                f" list or tuple. Received: {type(std)}"
            )

        if isinstance(anchor_shape, (list, tuple)):
            anchor_shape = np.array(anchor_shape)
        elif  anchor_shape and not isinstance(anchor_shape, np.ndarray):
            raise ValueError(
                "Expected anchor_shape to be a np.ndarray,"
                f" list or tuple. Received: {type(std)}"
            )

        self.mean = mean
        self.std = std
        self.anchor_shape = anchor_shape

        super(RegressBoxes, self).__init__(*args, **kwargs)

        if anchor_shape is not None:
            self.anchors = self.add_weight(
                name="anchor_boxes_baked",
                dtype="float32",
                shape=anchor_shape,
                initializer="ones",
                trainable=False,
            )

    def call(self, inputs, **kwargs):
        """
        Inputs contains an array of anchors boxes and array of predicted
        deltas from those anchor boxes.

        inputs has shape (2, B, N, 4).

        2=anchors and predicted deltas. 
        B=batch_size
        N=is the number of predicted boxes for each image
        4=box coordinates   
        """
        if self.saved_anchors:
            anchors = self.anchors
            predicted_deltas = inputs[0]
        else:
            anchors, predicted_deltas = inputs

        return apply_bbox_deltas(
            anchors, predicted_deltas, mean=self.mean, std=self.std
        )

    def compute_output_shape(self, input_shape):
        if self.saved_anchors:
            return self.anchors
        elif len(input_shape) == 2:
            return input_shape[0]
        else:
            raise ValueError(
                self.__class__,
                "needs to either get anchors as an input"
                "or set before use with set_anchors()",
            )

    def set_anchors(self, anchors: np.array):
        """
        Save anchor boxes as layer weights.

        This is a meaningfuly named wrapper around set_weights().
        """
        self.saved_anchors = True
        self.set_weights([anchors.astype("float32")])

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update(
            {
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "anchor_shape": self.anchor_shape.tolist(),
            }
        )

        return config


def apply_bbox_deltas(
    boxes: np.array, deltas: np.array, mean=default_mean, std=default_std
) -> np.array:
    """
    Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was
    previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator.
    They are unnormalized in this function and then applied to the boxes.

    :param np.array boxes: np.array of shape (B, N, 4), where B is the batch
         size, N the number of boxes and 4 values for (x1, y1, x2, y2).
    :param np.array deltas: np.array of same shape as boxes.
        These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
    :param list mean: The mean value used when computing deltas
        (defaults to [0, 0, 0, 0]).
    :param list std: The standard deviation used when computing deltas

    :return: A np.array of the same shape as boxes, but with deltas applied to
        each box. The mean and std are used during training to normalize the
        regression values (networks love normalization).
    :rtype: np.array
    """

    # subtract x2s from x1s
    widths = boxes[:, :, 2] - boxes[:, :, 0]
    # subtract y2s from y1s
    heights = boxes[:, :, 3] - boxes[:, :, 1]

    # stack these to multiply them by [x1, y1, x2, y2]
    scales = keras.backend.stack([widths, heights, widths, heights], axis=2)

    normalized_deltas = deltas[:, :] * std + mean

    # scale normalized deltas by their dimension size and add to anchors
    bboxes = boxes[:, :] + normalized_deltas * scales

    return bboxes
