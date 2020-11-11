import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

import anchors as anchor_functions
from model import image_sizes

import utils
import anchors_tpu


parser = argparse.ArgumentParser(
    description='Generate a preprocessed dataset .')

parser.add_argument('--dataset', default="voc",
                    help='Train on this stock dataset. See: https://www.tensorflow.org/datasets/catalog/voc')

parser.add_argument(
    '--workers', help='Number of multiprocessing workers. Defaults to autotune.', type=int,
    default=tf.data.experimental.AUTOTUNE)

parser.add_argument(
    '--num_shards', help='Number of shards to split data', type=int, default=100)

parser.add_argument(
    '--debug', help='Make assertions about inputs, check for nans', action='store_true')

parser.add_argument('--path', type=str, help='out path')

parser.add_argument('--phi', help='Hyper parameter phi',
                    default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))

parser.add_argument(
    '--split', type=str, help='the dataset split. Must be train, test, all, or validation')


def preprocess(args):
    split = getattr(tfds.Split, args.split.capitalize())
    info = tfds.builder(args.dataset).info

    image_size = image_sizes[args.phi]
    image_shape = (image_size, image_size)
    anchors = anchor_functions.anchors_for_shape(image_shape)

    num_classes = info.features['labels'].num_classes
    num_shards = args.num_shards

    labels_target_shape = (anchors.shape[0], num_classes + 1)
    regression_target_shape = (anchors.shape[0], 4 + 1)
    scaled_image_shape = (image_size, image_size, 3)

    dataset = tfds.load(args.dataset, split=split)

    dataset = dataset.enumerate().apply(tf.data.experimental.group_by_window(
        lambda i, _: i % num_shards, reduce_func, tf.int64.max
    ))

    def reduce_func(key, dataset):
        filename = tf.strings.join([args.path, tf.strings.as_string(key)])
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(dataset.map(lambda _, x: x))
        return tf.data.Dataset.from_tensors(filename)

    def preprocess_one_np(image_tensor, bboxes_tensor, labels):
        # input bboxes are an array of (n, 4)
        # where the format is ymin,xmin,ymax,xmax
        # with relative coords between 0 and 1.
        image = image_tensor.numpy()
        bboxes = bboxes_tensor.numpy()
        labels = labels.numpy()

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
        objects = x['objects']
        bboxes = objects["bbox"]
        labels = objects["label"]

        scaled_image, regression_target, labels_target = tf.numpy_function(
            preprocess_one_np, [image, bboxes, labels], Tout=(
                tf.float32, tf.float32, tf.float32),
            name="preprocessExample"
        )

        # These need to be set because tf cannot infer it
        labels_target.set_shape(labels_target_shape)
        regression_target.set_shape(regression_target_shape)
        scaled_image.set_shape(scaled_image_shape)

        return scaled_image, (regression_target, labels_target)


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

    return image, (regression_target, labels_target)


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess(args)
