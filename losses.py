"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import keras
from tensorflow import keras
import tensorflow as tf


def tpu_focal(alpha=0.25, gamma=2.0):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        # Use the cpu because using tf.where to get
        # indeces is not supported on TPU
        tf.debugging.check_numerics(y_pred, "ytrue in focal")
        tf.debugging.check_numerics(y_true, "ytrue in focal")

        labels = y_true[:, :, :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.compat.v1.where(tf.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = tf.ones_like(labels) * alpha

        foreground = tf.equal(labels, 1)

        alpha_factor = tf.compat.v1.where(foreground,
                                x=alpha_factor,
                                y=1 - alpha_factor)

        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.compat.v1.where(foreground,
                                x=1 - classification,
                                y=classification)

        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * \
            keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.compat.v1.where(keras.backend.equal(anchor_state, 1))
        normalizer = tf.cast(tf.shape(input=normalizer)[0], tf.float32)
        normalizer = tf.maximum(
            keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        x = keras.backend.equal(anchor_state, 1)
        indices = tf.compat.v1.where(x)

        # THIS DOES NOT WORK becuse where is broke
        # tf.keras
        # regression = tf.gather_nd(regression, indices)
        # regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.compat.v1.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(
            1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(
            normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

        return _smooth_l1


def tpu_smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        # (-1 for ignore, 0 for bg, 1 for fg)
        # indices = tf.where(x)

        # THIS DOES NOT WORK becuse where is broke
        # tf.keras
        # regression = tf.gather_nd(regression, indices)
        # regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target

        # Make every loss 0 for background and ignore
        condition = tf.equal(anchor_state, 1)

        # regression_diff = tf.where(
        #     tf.broadcast_to(condition, tf.shape(regression_diff)),
        #     x=regression_diff, y=tf.zeros_like(regression_diff),
        #     name="foreground_filter")

        maskf = tf.cast(condition, tf.float32)
        mask = tf.stack([maskf, maskf,  maskf, maskf], axis=2)

        regression_diff = tf.multiply(regression_diff, mask)

        # Treat False as 0

        regression_diff = tf.math.abs(regression_diff)

        regression_loss = tf.compat.v1.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = tf.maximum(1, tf.shape(input=regression_loss)[0])

        normalizer = tf.cast(normalizer, tf.float32)

        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1
