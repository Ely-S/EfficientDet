
import tensorflow as tf

from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES


EFFICIENTNET_DEPTHS = [227, 329, 329, 374, 464, 566, 656]


def load_weights(model, weights: str, phi: int, log):
    if weights:
        log.debug('Loading weights from %s, this may take a second...',
                  weights)

        if weights.endswith(".h5"):
            model.load_weights(weights, by_name=True)
        else:
            # Topological loading is only available in SavedModel format
            restore = model.load_weights(weights, by_name=False)
            restore.expect_partial()

        log.debug("done loading weights")
    else:
        log.debug("Downloading Weights")
        model_name = 'efficientnet-b{}'.format(phi)
        file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(
            model_name)
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = tf.keras.utils.get_file(file_name,
                                               BASE_WEIGHTS_PATH + file_name,
                                               cache_subdir='models',
                                               file_hash=file_hash)
        log.debug("Loading weights")
        model.load_weights(weights_path, by_name=True)
        log.debug("Loaded weigths")


def check_values(image, labels):
    regression_target, labels_target = labels

    tf.compat.v1.debugging.assert_all_finite(
        image, "Image contains NaN or Infinity")

    tf.compat.v1.debugging.assert_type(
        image, tf.float32,
        message="Input image must be a float32")

    tf.compat.v1.debugging.assert_all_finite(
        regression_target, "Regression target contains NaN or Infinity")

    tf.compat.v1.debugging.assert_type(
        regression_target, tf.float32,
        message="input regression boxes must be a float32")

    tf.compat.v1.debugging.assert_all_finite(
        labels_target, "Labels contains Nan or Infinity")

    tf.compat.v1.assert_greater_equal(
        labels_target[:, :-1], tf.cast(0, tf.float32),
        summarize=100,
        message="class labels are not everywhere >=0")

    return image, labels


class CheckpointSaver(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_prefix, log=None):
        super(CheckpointSaver, self).__init__()
        self.checkpoint_prefix = checkpoint_prefix
        self.log = log

    def on_train_begin(self, logs):
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.model.optimizer,
            model=self.model)

    def on_epoch_end(self, epoch, logs):
        # model.save_wieghts also works
        file_prefix = self.checkpoint_prefix.format(epoch=epoch, **logs)

        if self.log:
            self.log.debug("saving to %s", file_prefix)

        # model.save_wieghts cannot be loaded by a Checkpoint.
        # model.load_weigths does not work with the distriution strategy
        # .write is more reliable than .save, which
        # fights the distribution scope

        self.checkpoint.write(file_prefix)

        if self.log:
            self.log.debug("saved %s", file_prefix)
