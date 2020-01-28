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

EFFICIENTNET_DEPTHS = [227, 329, 329, 374, 464, 566, 656]


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


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

    parser.add_argument(
        '--workers', help='Number of multiprocessing workers', type=int, default=1)

    parsed_args = parser.parse_args()
    print(parsed_args)
    return parsed_args


def load_weights(model, snapshot, phi):
    if snapshot:
        print('Loading weights, this may take a second...')
        model.load_weights(snapshot, by_name=True)
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


def train_model(model, epochs, input_train_data, input_eval_data):
    model_dir = "/tmp/"

    estimator = tf.keras.estimator.model_to_estimator(
        model,
        model_dir=model_dir,
    )

    for i in range(0, epochs):
        print("Epoch", i)

        estimator.train(input_train_data)

        print("eval")
        estimator.evaluate(
            input_eval_data,
            hooks=None,
            checkpoint_path=None,
            name="val"
        )


def main(args=None):
    tf.keras.backend.set_session(get_session())

    # tfds works in both Eager and Graph modes
    # tf.compat.v1.enable_eager_execution()

    # .prefetch(tf.data.experimental.AUTOTUNE)

    workers = args.workers
    batch_size = args.batch_size
    datasetName = args.dataset
    image_size = (image_sizes[args.phi], image_sizes[args.phi])

    anchors = utils.anchors.anchors_for_shape(image_size)

    info = tfds.builder(datasetName).info
    num_classes = info.features['label'].num_classes

    def preprocess_one(x):
        image = x['image'].numpy()
        objects = x['objects']

        # input bboxes are an array of (n, 4)
        # where the format is ymin,xmin,ymax,xmax
        # with relative coords between 0 and 1.
        bboxes = objects['bbox'].numpy()

        height, width, channels = image.shape

        # The model expects absolute bbox coords
        # in the order [x1, y1, x2, y2] not [y1, x1, y2, x2]
        bboxes = bboxes[:, [1, 0, 3, 2]]

        # Change from relative to absolute coords
        bboxes[:, [0, 2]] *= width
        bboxes[:, [1, 3]] *= height

        scaled_image, scale, offset_h, offset_w = utils.preprocess_image(
            image, image_size)

        # apply resizing to annotations too
        bboxes *= scale
        bboxes[:, [0, 2]] += offset_w
        bboxes[:, [1, 3]] += offset_h

        annotations = {
            "label": objects["label"],
            "bboxes": bboxes
        }

        return scaled_image, annotations

    def preprocess_batch(image_batch, annotations_batch):
        targets = anchors.anchor_targets_bbox(
            anchors,
            image_batch,
            annotations_batch,
            num_classes
        )

        return image_batch, targets

    def input_train_data():
        split = tfds.Split.TRAIN
        dataset = tfds.load(datasetName, split=split, as_supervised=True)
        batch = dataset.map(preprocess_one, num_parallel_calls=workers
                            ).batch(batch_size
                                    ).map(preprocess_batch, num_parallel_calls=workers)
        return batch

    def input_eval_data():
        split = tfds.Split.VALIDATION
        dataset = tfds.load(datasetName, split=split, as_supervised=True)
        batch = dataset.map(preprocess_one, num_parallel_calls=workers
                            ).batch(batch_size
                                    ).map(preprocess_batch, num_parallel_calls=workers)
        return batch

    model, _ = efficientdet(args.phi,
                            num_classes=num_classes,
                            anchors=anchors,
                            weighted_bifpn=args.weighted_bifpn,
                            freeze_bn=args.freeze_backbone)

    load_weights(model, args.snapshot, args.phi)

    # freeze backbone layers
    if args.freeze_backbone:
        for i in range(1, EFFICIENTNET_DEPTHS[args.phi]):
            model.layers[i].trainable = False

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

    # compile model
    model.compile(optimizer=optimizer, loss={
        'regression': smooth_l1(),
        'classification': focal_loss
    })

    train_model(model, args.epochs, input_train_data, input_eval_data)


if __name__ == '__main__':
    import initializers
    scope = {
        "PriorProbability": initializers.PriorProbability
    }

    with tf.keras.utils.custom_object_scope(scope):
        main(parse_args())
