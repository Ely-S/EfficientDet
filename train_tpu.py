import argparse
from datetime import date
import os
import sys
import math
import logging

import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

import utils.anchors
from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect
from model import efficientdet, image_sizes
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

import utils
import utils.anchors_tpu
import utils.tpu as tpu

tf.debugging.set_log_device_placement(True)
log = logging.getLogger(__file__)

EFFICIENTNET_DEPTHS = [227, 329, 329, 374, 464, 566, 656]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an EfficientDet model on a TPU.')

    parser.add_argument(
        '--verbose', help='Keras fit_generator verbose setting', default=1, type=int)

    parser.add_argument('--phi', help='Hyper parameter phi',
                        default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--weighted-bifpn', help='Use weighted BiFPN',
                        action='store_true')

    parser.add_argument('--dataset', default="voc",
                        help='Train on this stock dataset. See: https://www.tensorflow.org/datasets/catalog/voc')

    parser.add_argument('--weights', default=None,
                        help='Initialize weights using file.')

    parser.add_argument(
        '--epochs', help='Number of epochs to train. An epoch is a pass over the whole dataset', type=int, default=50)

    parser.add_argument(
        '--batch-size', help='Size of the batch in each pass.', default=32, type=int)

    parser.add_argument(
        '--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument(
        '--freeze-batchnorm', help='Freeze training of BatchNormalization layers.', action='store_true')

    parser.add_argument("--weighted", dest="weighted_bifpn",
                        action="store_true")

    parser.add_argument(
        '--workers', help='Number of multiprocessing workers. Defaults to autotune.',
        type=int,
        default=tf.data.experimental.AUTOTUNE)

    parser.add_argument(
        '--debug', help='Make assertions about inputs, check for nans', action='store_true')

    parsed_args = parser.parse_args()
    return parsed_args


def load_weights(model, weights, phi):
    if weights:
        log.debug('Loading weights, this may take a second...')
        model.load_weights(weights, by_name=True)
        log.debug("done loading model")
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


def train_model(model, epochs,  steps_per_epoch,
                validation_steps, train_data, val_data):
    # The Optimizer
    # See https://github.com/shaoanlu/dogs-vs-cats-redux/blob/master/opt_experiment.ipynb
    # for benchmark of optimizers.
    # @TODO: Tune this
    # @TODO: Add Exponential Decay
    # @TODO: Add Early Stopping
    # @TODO: Add Warmup
    optimizer = tf.keras.optimizers.SGD(lr=.08, decay=4e-5, momentum=0.9)

    # alpha and gamma values come from the EfficientDet paper
    classification_loss = tpu.tpu_focal(alpha=0.25, gamma=1.5)
    regression_loss = tpu.tpu_smooth_l1()

    # note: tfa is not supported by tf 1.15.0
    # focal_loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=.25, gamma=1.5)

    log.debug("Compiling model")

    # Work around for: https://github.com/tensorflow/tensorflow/issues/34199
    model.compile(
        optimizer=optimizer,
        loss=[regression_loss, classification_loss])

    # model.compile(optimizer=optimizer, loss={
    #     'classification': ,
    #     'regression': ,
    # })

    log.debug("Fitting model")

    history = model.fit(train_data,
                        verbose=2,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_data,
                        validation_steps=validation_steps,
                        epochs=epochs)


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


def main(args=None):
    workers = args.workers
    batch_size = args.batch_size
    datasetName = args.dataset

    image_size = image_sizes[args.phi]
    image_shape = (image_size, image_size)

    info = tfds.builder(datasetName).info
    num_classes = info.features['labels'].num_classes

    anchors = utils.anchors.anchors_for_shape(image_shape)
    labels_target_shape = tf.TensorShape((anchors.shape[0], num_classes + 1))
    regression_target_shape = tf.TensorShape((anchors.shape[0], 4 + 1))
    scaled_image_shape = tf.TensorShape((image_size, image_size, 3))

    image_feature_description = {
        # PNGs may be compresed to different lengths
        'image': tf.io.FixedLenFeature([], tf.string),
        'regression': tf.io.FixedLenFeature([], tf.string),
        'label':  tf.io.FixedLenFeature([], tf.string),
    }

    def parse_example_file(serialized_example, *args):
        # example = tf.train.Example.ParseFromString(tfrecord)

        example = tf.io.parse_single_example(serialized_example,
                                             image_feature_description)

        # PNGs are represented as uints, which is why they are unscaled
        png = tf.io.decode_png(example["image"])

        regression_target = tf.io.parse_tensor(
            example["regression"], out_type=tf.float32)
        label_target = tf.io.parse_tensor(
            example["label"], out_type=tf.float32)

        image = tf.cast(png, tf.float32) / 255

        label_target.set_shape(labels_target_shape)
        regression_target.set_shape(regression_target_shape)
        image.set_shape(scaled_image_shape)

        return image, (regression_target, label_target)

    def input_data(global_batch_size, folder):
        files = tf.data.Dataset.list_files(os.path.join(folder, "*.tfrecord"))

        dataset = tf.data.TFRecordDataset(
            files,
            num_parallel_reads=workers)

        # Shuffle before repeat so that every record appears
        # in every batch
        dataset = dataset.shuffle(1000)

        # Keras expects a generator that lasts forever
        dataset = dataset.repeat()

        dataset = dataset.map(parse_example_file, workers)

        if args.debug:
            dataset = dataset.map(check_values)

        # drop_remainder is important on TPU, batch size must be fixed
        dataset = dataset.batch(global_batch_size, drop_remainder=True)

        # Load next batch before it is needed
        prefetched = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return prefetched

    # number is samples in voc training
    steps_per_epoch = math.ceil(2501 / batch_size)
    # number is samples in voc validation
    steps_per_val_epoch = math.ceil(2501 / batch_size)

    distribution_strategy = tpu.get_strategy()

    tf.debugging.set_log_device_placement(True)

    with start_session():

        log.debug("Creating model")
        local_model = efficientdet(
            args.phi,
            num_classes=num_classes,
            just_training_model=True,
            # anchors=anchors,
            weighted_bifpn=args.weighted_bifpn,
            freeze_bn=args.freeze_backbone)

        log.debug("Created Model")

        load_weights(local_model, args.weights, args.phi)

        # freeze backbone layers
        if args.freeze_backbone:
            for i in range(1, EFFICIENTNET_DEPTHS[args.phi]):
                local_model.layers[i].trainable = False

        saved_model_path = "/tmp/tf_save"
        log.debug("Saving temp model to %s", saved_model_path)
        tf.saved_model.save(local_model, saved_model_path)
        log.debug("saved model")

    log.debug("Entering Distribution Strategy")
    with distribution_strategy.scope():
        # Clear memory from the previous model
        with start_session():

            tf.debugging.set_log_device_placement(True)

            log.debug("Loading model across cluster")
            model = tf.keras.models.load_model(saved_model_path)
            log.debug("loaded model")

            if args.debug:
                physical_devices = tf.config.experimental_list_devices()
                log.debug("Processors %s", physical_devices)
                tf.debugging.set_log_device_placement(True)

            global_batch_size = (batch_size *
                                 distribution_strategy.num_replicas_in_sync)

            train_data = input_data(
                global_batch_size, "gs://ondaka-ml-data/dev/run1/train")
            val_data = input_data(
                global_batch_size, "gs://ondaka-ml-data/dev/run1/val")

            log.debug("Starting to train")
            train_model(model, args.epochs, steps_per_epoch,
                        steps_per_val_epoch, train_data, val_data)


def start_session():
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config, graph=tf.Graph())
    tf.compat.v1.keras.backend.set_session(session)
    return session


if __name__ == '__main__':
    import initializers

    args = parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
        tf.debugging.set_log_device_placement(True)
        log.debug("args %s", args)

    object_scope = tf.keras.utils.custom_object_scope({
        "PriorProbability": initializers.PriorProbability
    })

    with object_scope:
        main(args)
        log.info("Done")
