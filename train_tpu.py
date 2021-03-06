import glob
import os
import sys
import math
import logging
import pathlib
import argparse
from datetime import date

import tensorflow as tf
import tensorflow_datasets as tfds
import h5py

from utils.lr_schedule import get_cosine_decay_with_linear_warmup
from model import efficientdet, image_sizes
import utils.tpu as tpu
import utils.anchors
import utils

from utils.train import load_weights, check_values, CheckpointSaver

log = logging.getLogger(__file__)

EFFICIENTNET_DEPTHS = [227, 329, 329, 374, 464, 566, 656]

tf.compat.v1.disable_eager_execution()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an EfficientDet model on a TPU.')

    parser.add_argument(
        '--verbose', help='Keras fit_generator verbose setting', default=1, type=int)

    parser.add_argument('--phi', help='Hyper parameter phi',
                        default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--weighted-bifpn', help='Use weighted BiFPN',
                        action='store_true')

    parser.add_argument('--dataset',
                        help='A directory with train/ and val/ sub dirs. Can be a gs:// url.')

    parser.add_argument('--num-classes', default=20, type=int,
                        help='The number of classes to train for. Default is 20 for VOC.')

    parser.add_argument('--log-dir', type=str,
                        help='Destination for Tensorboard logs. '
                             'Should start with gs:// for TPU training')

    parser.add_argument('--weights', default=None,
                        help='Initialize weights using file.')

    parser.add_argument(
        '--epochs', help='Number of epochs to train. An epoch is a pass over the whole dataset', type=int, default=50)

    parser.add_argument("--initial-epoch", default=0, type=int)

    parser.add_argument(
        '--batch-size', help='Size of the batch per device in each pass. 12 is good size for a 16GB V100 GPU.', default=32, type=int)

    parser.add_argument(
        '--restore',
        help='Restore training from latest checkpoint',
        action="store_true")

    parser.add_argument('--cache', default=False,
                        action="store_true",
                        help='Cache TF records locally')

    parser.add_argument(
        '--cache-dir',
        default="",
        help='Cache tfrecords in this dir. If --cache is given '
        'and this argument left blank,  cache is stored in memory.')

    parser.add_argument("--q-size",
                        type=int,
                        help="Prefetch this number of input examples. Higher numbers "
                             "Increases memory usage.",
                        default=tf.data.experimental.AUTOTUNE)

    parser.add_argument(
        '--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument(
        '--freeze-batchnorm', help='Freeze training of BatchNormalization layers.', action='store_true')

    parser.add_argument(
        '--workers', help='Number of multiprocessing workers. Defaults to autotune.',
        type=int,
        default=tf.data.experimental.AUTOTUNE)

    parser.add_argument(
        "--shuffle",
        help="Number of input examples to shuffle. Higher values use more memory.",
        default=300,
        type=int
    )

    parser.add_argument(
        "--steps",
        help="steps per epoch",
        default=200,
        type=int
    )

    parser.add_argument(
        "--n-val",
        help="Number of validation examples. Validation"
             "steps is calculated as = n-val / batch-size",
        default=2500,
        type=int
    )
    parser.add_argument("--check-input", action="store_true", default=False)

    parser.add_argument('--pu', help="Type of processing unit to use. Defaults to 'tpu'",
                        default="tpu", type=str, choices=["tpu", "gpu", "cpu"])

    parser.add_argument(
        '--checkpoints',
        help='The folder to place checkpoints. This is created if it needed.',
        default='checkpoints')

    parser.add_argument(
        '--debug', help='Make assertions about inputs, check for nans', action='store_true')

    parsed_args = parser.parse_args()
    return parsed_args


mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])


def normalize_image(image):
    """
    Normalize image by subtracting and dividing by the pre-computed
    image-net mean and std. dev.
    """
    return ((image/255.0) - mean) / std


def main(args=None):
    workers = args.workers
    batch_size = args.batch_size

    image_size = image_sizes[args.phi]
    image_shape = (image_size, image_size)

    num_classes = args.num_classes

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

        example = tf.io.parse_single_example(serialized=serialized_example,
                                             features=image_feature_description)

        # PNGs are represented as uints, which is why they are unscaled
        png = tf.io.decode_png(example["image"])

        regression_target = tf.io.parse_tensor(
            example["regression"], out_type=tf.float32)
        label_target = tf.io.parse_tensor(
            example["label"], out_type=tf.float32)

        label_target.set_shape(labels_target_shape)
        regression_target.set_shape(regression_target_shape)
        png.set_shape(scaled_image_shape)

        image = tf.cast(png, tf.float32)
        normalized_image = normalize_image(image)

        return normalized_image, (regression_target, label_target)

    def input_data(global_batch_size, folder, validation=False):
        files = tf.data.Dataset.list_files(os.path.join(folder, "*.tfrecord"))

        dataset = tf.data.TFRecordDataset(
            files,
            num_parallel_reads=workers)

        if args.cache:
            if args.cache_dir != "":
                cache_dir = pathlib.Path(args.cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)

                # There should be seperate cache files for
                #  train, test, and validation sets
                cacheFile = str(cache_dir / folder.replace("/", "_"))

                dataset = dataset.cache(cacheFile)
            else:
                # Cache in memory
                dataset = dataset.cache()

        if not validation:
            # Shuffle before repeat so that every record appears
            # equally in every batch. This is unnecessary for validation
            # Which should include every validation example.
            dataset = dataset.shuffle(args.shuffle)

        # Keras expects a generator that lasts forever
        dataset = dataset.repeat()

        # Decode the images here to avoid buffering raw image data
        dataset = dataset.map(parse_example_file, workers)

        if args.check_input:
            dataset = dataset.map(check_values)

        # Load next item before it is needed
        dataset = dataset.prefetch(args.q_size)

        # batch size must be fixed for TPU, so drop_remainder=True
        dataset = dataset.batch(global_batch_size, drop_remainder=True)

        return dataset

    # Validation size
    steps_per_val_epoch = math.ceil(args.n_val / batch_size)

    if args.pu == "tpu":
        distribution_strategy = tpu.get_strategy()
    elif args.pu == "cpu":
        distribution_strategy = tf.distribute.OneDeviceStrategy('/CPU')
    elif args.pu == "gpu":
        distribution_strategy = tf.distribute.MirroredStrategy()
    else:
        raise ValueError("%s Not a valid processing unit", args.pu)

    log.debug("Entering Distribution Strategy")

    start_session()
    scope = distribution_strategy.scope()

    with scope:

        model = efficientdet(
            args.phi,
            num_classes=num_classes,
            just_training_model=True,
            weighted_bifpn=args.weighted_bifpn,
            freeze_bn=args.freeze_batchnorm)

        log.debug("Built model")

        # alpha and gamma values come from the EfficientDet paper
        classification_loss = tpu.tpu_focal(alpha=0.25, gamma=1.5)
        regression_loss = tpu.tpu_smooth_l1()

        # The Optimizer
        # See https://github.com/shaoanlu/dogs-vs-cats-redux/blob/master/opt_experiment.ipynb
        # for benchmark of optimizers.
        #
        # These values come from the paper.
        log.debug("Compiling model")
        optimizer = tf.keras.optimizers.SGD(
            lr=0.01, decay=4e-5, momentum=0.9)

        # freeze backbone layers
        if args.freeze_backbone:
            for i in range(1, EFFICIENTNET_DEPTHS[args.phi]):
                model.layers[i].trainable = False

        model.compile(
            optimizer=optimizer,
            loss={
                'classification': classification_loss,
                'regression': regression_loss,
            },
        )

        if args.restore:
            checkpoints = glob.glob(os.path.join(args.checkpoints, "*.index"))

            if len(checkpoints) == 0:
                raise ValueError(
                    "--restore option given but no previous "
                    "checkpoint found to restore from.")

            latest = max(checkpoints, key=os.path.getctime
                         ).replace(".index", "")

            # if index is included, then it assumes a v1 checkpoint
            # tf.train.latest_checkpoint doesnt work because it assumes
            # .write was used to create the checkpoint

            log.debug("Restoring checkpoint %s", latest)
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            status = checkpoint.restore(latest)

            # the optimizer weights are not yet created
            status.expect_partial()

        elif args.restore and args.weights:
            raise ValueError(
                "--restore and --weights should not both be present.")
        else:
            load_weights(model, args.weights, args.phi, log)

        log.debug("Number of devices: %s",
                  distribution_strategy.num_replicas_in_sync)

        global_batch_size = (batch_size *
                             distribution_strategy.num_replicas_in_sync)

        log.debug("Global Batch Size %s", global_batch_size)

        train_dir = os.path.join(args.dataset, "train")
        val_dir = os.path.join(args.dataset, "val")

        train_data = input_data(global_batch_size, train_dir)
        val_data = input_data(global_batch_size, val_dir, validation=True)

        callbacks = get_callbacks(args)

        # Fitting happens outside of scope
        log.debug("Fitting model")
        history = model.fit(
            train_data,

            # Saving model does not work inside of scoe
            callbacks=callbacks,

            # Print porgress bar
            verbose=1,

            epochs=args.epochs,
            steps_per_epoch=args.steps,
            initial_epoch=args.initial_epoch,

            # Note: Validation does not seem to work on tpus
            validation_steps=steps_per_val_epoch,
            validation_data=val_data,
        )


def get_callbacks(args):
    checkpoint_dir = pathlib.Path(args.checkpoints)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = str(
        checkpoint_dir /
        "checkpoint_{epoch}_{val_loss}_{val_regression_loss}_{val_classification_loss}")

    # This is the schedule described in the paper
    warmup_and_decay_lr = get_cosine_decay_with_linear_warmup(
        learning_rate_start=0.05,
        total_epochs=args.epochs,
        verbose=True)

    log.debug("Saving checkpoints as '%s'", checkpoint_prefix)

    callbacks = [
        warmup_and_decay_lr,
        CheckpointSaver(checkpoint_prefix, log),
    ]

    if args.log_dir:
        tb = tf.keras.callbacks.TensorBoard(
            log_dir=args.log_dir,
        )
        callbacks.append(tb)

    return callbacks


def start_session(target=''):
    config = tf.compat.v1.ConfigProto()
    session = tf.compat.v1.Session(target, config=config)
    tf.compat.v1.keras.backend.set_session(session)
    return session


if __name__ == '__main__':
    import initializers

    args = parse_args()

    if args.debug:
        tf.get_logger().setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)
        log.debug("args %s", args)

    object_scope = tf.keras.utils.custom_object_scope({
        "PriorProbability": initializers.PriorProbability
    })

    with object_scope:
        main(args)
        log.info("Done")
