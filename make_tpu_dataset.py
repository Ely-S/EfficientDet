import os
import math
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

import utils
import utils.anchors
import utils.anchors_tpu
import utils.compute_overlap
from model import image_sizes


parser = argparse.ArgumentParser(
    description='Generate a preprocessed dataset .')

parser.add_argument('--dataset', default="voc",
                    help='Train on this stock dataset. See: https://www.tensorflow.org/datasets/catalog/voc')

parser.add_argument(
    '--workers', help='Number of multiprocessing workers. Defaults to autotune.', type=int,
    default=tf.data.experimental.AUTOTUNE)

parser.add_argument(
    '--num-shards', help='Number of shards to split data', type=int, default=100)

parser.add_argument(
    '--debug', help='Make assertions about inputs, check for nans', action='store_true')

parser.add_argument('--path', required=True, type=str, help='out path')

parser.add_argument('--phi', help='Hyper parameter phi',
                    required=True,
                    type=int, choices=(0, 1, 2, 3, 4, 5, 6))

parser.add_argument(
    '--split', type=str, required=True,
    choices=("train", "test", "validation", "all"),
    help='the dataset split. Must be train, test, all, or validation')

tf.compat.v1.enable_eager_execution()


def preprocess(args):
    image_size = image_sizes[args.phi]
    image_shape = (image_size, image_size)
    anchors = utils.anchors.anchors_for_shape(image_shape)

    split = getattr(tfds.Split, args.split.upper())
    info = tfds.builder(args.dataset).info
    num_classes = info.features['labels'].num_classes
    num_examples = info.splits[split].num_examples

    num_shards = args.num_shards
    examples_per_shard = math.ceil(num_examples / num_shards)

    print("Examples per shard", examples_per_shard)

    labels_target_shape = tf.TensorShape((anchors.shape[0], num_classes + 1))
    regression_target_shape = tf.TensorShape((anchors.shape[0], 4 + 1))
    scaled_image_shape = tf.TensorShape((image_size, image_size, 3))

    def preprocess_one_np(image, bboxes, labels):
        # input bboxes are an array of (n, 4)
        # where the format is ymin,xmin,ymax,xmax
        # with relative coords between 0 and 1.
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

        regression_target, labels_target = utils.anchors_tpu.gen_anchor_targets(
            anchors, scaled_image, bboxes, labels, num_classes)

        return scaled_image, regression_target, labels_target

    def preprocess_one(x):
        image = x['image']
        bboxes = x['objects']["bbox"]
        labels = x['objects']["label"]

        scaled_image, regression_target, labels_target = tf.numpy_function(
            preprocess_one_np, [image, bboxes, labels], Tout=(
                tf.float32, tf.float32, tf.float32),
            name="preprocessExample"
        )

        # These need to be set because tf cannot infer it
        labels_target.set_shape(labels_target_shape)
        regression_target.set_shape(regression_target_shape)
        scaled_image.set_shape(scaled_image_shape)

        # scale these back up because they will be stored as uints
        rescaled_image = scaled_image * 255

        return rescaled_image, regression_target, labels_target

    dataset = tfds.load(args.dataset, split=split)
    dataset = dataset.map(preprocess_one, args.workers)
    dataset = dataset.map(check_values)
    batches = dataset.batch(examples_per_shard, drop_remainder=False,)
    batches = batches.prefetch(1)

    for i, batch in enumerate(batches):
        path = os.path.join(args.path, "{i:5d}.tfrecord".format(i=i))
        print("Writing", path)
        write_batch(batch, path)

    print("Done")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_batch(batch, path):
    with tf.io.TFRecordWriter(path) as writer:
        for (image, regression, label) in zip(*batch):
            # PNGs have to be uints
            image = tf.cast(image, tf.uint8)

            feature = {
                "image": _bytes_feature(tf.image.encode_png(image)),
                "regression": _bytes_feature(tf.io.serialize_tensor(regression)),
                "label": _bytes_feature(tf.io.serialize_tensor(label))
            }

            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature))

            serialized_example = example_proto.SerializeToString()

        writer.write(serialized_example)


def check_values(image, regression_target, labels_target):
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

    return image, regression_target, labels_target


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess(args)
