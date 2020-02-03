import argparse
from datetime import date
import os
import sys
import math

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import cv2

import utils.anchors
from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect
from model import efficientdet, image_sizes
from losses import smooth_l1, focal
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

import utils
import utils.anchors_tpu
import utils.tpu as tpu

tf.debugging.set_log_device_placement(True)

EFFICIENTNET_DEPTHS = [227, 329, 329, 374, 464, 566, 656]

# tf.enable_v2_behavior()
# tf.compat.v1.enable_eager_execution()


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
        '--workers', help='Number of multiprocessing workers. Defaults to autotune.', type=int,
        default=tf.data.experimental.AUTOTUNE)

    parser.add_argument(
        '--debug', help='Make assertions about inputs, check for nans', action='store_true')

    parsed_args = parser.parse_args()
    return parsed_args


def load_weights(model, weights, phi):
    if weights:
        print('Loading weights, this may take a second...')
        model.load_weights(weights, by_name=True)
        print("done loading model")
    else:
        model_name = 'efficientnet-b{}'.format(phi)
        file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(
            model_name)
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = keras.utils.get_file(file_name,
                                            BASE_WEIGHTS_PATH + file_name,
                                            cache_subdir='models',
                                            file_hash=file_hash)
        model.load_weights(weights_path, by_name=True)


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
    focal_loss = focal(alpha=0.25, gamma=1.5)

    # note: tfa is not supported by tf 1.15.0
    # focal_loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=.25, gamma=1.5)

    model.compile(optimizer=optimizer, loss={
        'regression': smooth_l1(),
        'classification': focal_loss
    })

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

    tf.compat.v1.debugging.assert_all_finite(
        regression_target, "Regression target contains NaN or Infinity")

    tf.compat.v1.debugging.assert_all_finite(
        labels_target, "Labels contains Nan or Infinity")

    tf.compat.v1.assert_greater_equal(
        labels_target[:, :-1], 0.0,
        summarize=100,
        message="class labels are not everywhere >=0")

    return image, labels


def main(args=None):
    workers = args.workers
    batch_size = args.batch_size
    datasetName = args.dataset

    image_size = image_sizes[args.phi]
    image_shape = (image_size, image_size)

    anchors = utils.anchors.anchors_for_shape(image_shape)

    info = tfds.builder(datasetName).info
    num_classes = info.features['labels'].num_classes

    labels_target_shape = tf.TensorShape(
        (batch_size,  anchors.shape[0], num_classes + 1))
    regression_target_shape = tf.TensorShape(
        (batch_size, anchors.shape[0], 4 + 1))
    scaled_image_shape = tf.TensorShape(
        (batch_size, image_size, image_size, 3))

    def deserialize(features, *args):

        image_feature_description = {
            'image': tf.io.FixedLenFeature(scaled_image_shape, tf.float32),
            'regression': tf.io.FixedLenFeature(regression_target_shape, tf.float32),
            'label': tf.io.FixedLenFeature(labels_target_shape, tf.float32),
        }

        example = tf.io.parse_single_example(features,
                                             image_feature_description)

        print(example)
        image = example['image']
        regression = example['regression']
        label = example['label']

        return image, (regression, label)

    def input_data(global_batch_size, folder):
        files = tf.data.Dataset.list_files(os.path.join(folder, "*.tfrecord"))

        dataset = tf.data.TFRecordDataset(
            files,
            compression_type='ZLIB',
            num_parallel_reads=workers)

        # Keras expects a generator that lasts forever
        dataset = dataset.repeat()

        dataset = dataset.map(deserialize)

        # Remove the earlier batching
        dataset = dataset.unbatch()

        if args.debug:
            dataset = dataset.map(check_values)

        dataset = dataset.shuffle(1000)

        # drop_remainder is important on TPU, batch size must be fixed
        dataset = dataset.batch(global_batch_size, drop_remainder=True)

        # Load next batch before it is needed
        prefetched = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return prefetched

    # number is samples in voc training
    steps_per_epoch = math.ceil(2501 / batch_size)
    # number is samples in voc validation
    steps_per_val_epoch = math.ceil(2501 / batch_size)

    # distribution_strategy = tpu.get_strategy()

    distribution_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")

    with distribution_strategy.scope():

        global_batch_size = (batch_size *
                             distribution_strategy.num_replicas_in_sync)

        model = efficientdet(args.phi,
                             num_classes=num_classes,
                             just_training_model=True,
                             anchors=anchors,
                             weighted_bifpn=args.weighted_bifpn,
                             freeze_bn=args.freeze_backbone)

        print(100000000000 * 1)

        train_data = input_data(
            global_batch_size, "gs://ondaka-ml-data/dev/e1/train")
        val_data = input_data(
            global_batch_size, "gs://ondaka-ml-data/dev/e1/validation")

        load_weights(model, args.weights, args.phi)

        print(100000000000 * 2)

        # freeze backbone layers
        if args.freeze_backbone:
            for i in range(1, EFFICIENTNET_DEPTHS[args.phi]):
                model.layers[i].trainable = False

        print(100000000000 * 3)
        train_model(model, args.epochs, steps_per_epoch,
                    steps_per_val_epoch, train_data, val_data)


if __name__ == '__main__':
    import initializers
    scope = {
        "PriorProbability": initializers.PriorProbability
    }

    args = parse_args()

    if args.debug:
        tf.debugging.set_log_device_placement(True)
        physical_devices = tf.config.experimental_list_devices()
        print("Processors", physical_devices)
        print("args", args)

    with tf.keras.utils.custom_object_scope(scope):
        main(args)
