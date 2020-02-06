import tensorflow as tf

# The documentation on thsi takes some digigng to find, so here it is.
# See:https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy
# Example: https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/keras_mnist_tpu.ipynb
# Example2: https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/fashion_mnist.ipynb


def get_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        # coordinator_name='host'
    )

    tf.config.experimental_connect_to_cluster(resolver)

    tf.tpu.experimental.initialize_tpu_system(resolver)

    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    return strategy


def tpu_smooth_l1(lambda_=1):
    """
    Create a smooth L1 loss function. This is similar to huber loss.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    """

    def _smooth_l1(y_true, y_pred):
        """ Compute the Smooth L1 loss of bounding box predictions

        See: https://arxiv.org/pdf/1504.08083.pdf page 3 for a definition.


        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        with tf.compat.v1.name_scope("smooth_l1_loss"):
            regression_target = y_true[:, :, :-1]
            anchor_state = y_true[:, :, -1]

            regression_diff = tf.abs(y_pred - regression_target)

            regression_loss = tf.compat.v1.where(
                tf.greater(regression_diff, lambda_),
                x=regression_diff - 0.5,
                y=0.5 * (regression_diff ** 2),
            )

            # Exclude background and ignore anchors in the regression loss
            foreground_mask = tf.cast(
                tf.equal(anchor_state, 1), regression_loss.dtype)

            # The normalizer is the number of foreground anchors
            normalizer = tf.math.count_nonzero(
                foreground_mask, dtype=tf.float32)
            normalizer = tf.maximum(1.0, normalizer)

            # @TODO: Benchmark these two approaches to calculating total_loss
            # total_loss_ein = tf.einsum("ijk,ij", regression_loss, mask)

            per_anchor_loss = tf.math.reduce_sum(
                input_tensor=regression_loss, axis=2)

            foreground_anchor_loss = per_anchor_loss * foreground_mask

            total_loss = tf.math.reduce_sum(
                input_tensor=foreground_anchor_loss)

            return total_loss / normalizer

    return _smooth_l1


def tpu_focal(alpha=0.25, gamma=2.0):
    """
    return a function for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
    """
    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes+1).

        Returns
            The focal loss of y_pred w.r.t. y_true.

            Focal loss is defined as
            FL(p_t) =−αt(1−p_t)^γ * log(p_t)
        """
        with tf.compat.v1.name_scope("focal_loss"):
            # shape (B, N, num_classes). Excludes anchor_state
            true_class = y_true[:, :, :-1]
            pred_class = y_pred

            # anchor state is -1 for ignore, 0 for background, 1 for object
            anchor_state = y_true[:, :, -1]
            # anchor state shape is (B, N)

            # compute the focal loss
            _alpha = alpha + tf.zeros_like(true_class)

            is_foreground = tf.equal(true_class, 1)

            alpha_factor = tf.compat.v1.where(is_foreground,
                                              x=_alpha,
                                              y=1 - _alpha,
                                              name="alphafactor")

            # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
            focal_weight = tf.compat.v1.where(is_foreground,
                                              x=1 - pred_class,
                                              y=pred_class,
                                              name="alphaweight")

            focal_weight = alpha_factor * focal_weight ** gamma

            # focal_weight = ignore_
            cls_loss = focal_weight * \
                tf.keras.backend.binary_crossentropy(
                    true_class, pred_class)

            # Class loss is 0 for ignore anchors
            ignore_mask = tf.cast(
                tf.not_equal(anchor_state, -1), cls_loss.dtype)
            per_anchor_loss = tf.math.reduce_sum(input_tensor=cls_loss, axis=2)

            per_anchor_loss_without_ignore = per_anchor_loss * ignore_mask
            total_loss = tf.math.reduce_sum(
                input_tensor=per_anchor_loss_without_ignore)

            # compute the normalizer: the number of positive anchors
            positive_mask = tf.cast(tf.equal(anchor_state, 1), tf.int32)
            positive_count = tf.math.count_nonzero(
                positive_mask, dtype=tf.float32)
            normalizer = tf.maximum(1.0, positive_count)

            return total_loss / normalizer

    return _focal
