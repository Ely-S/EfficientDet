import numpy as np
import tensorflow as tf

import RegressBoxes

# dimension of this array is
# (batch size, num boxes, 4)
boxes = np.array(
    # Batch Size
    [
        # boxes
        [
            # 4 coords
            [0, 0, 1, 1],
            [0.5, 0.5, 0.6, 0.6],
        ]
    ],
).astype("float32")

# deltas is the same shape as boxes
# and during operation contains the predicted
# deltas from anchors
deltas = np.array(
    # batch size
    [
        # boxes
        [
            # coords
            [0.1, 0.1, 0.1, 0.1],
            [-0.2, -0.2, 0.2, 0.2],
        ]
    ]
).astype("float32")

outcome = np.array([[[0.02, 0.02, 1.02, 1.02], [0.496, 0.496, 0.604, 0.604]]])


def test_apply_bbox_deltas():
    """Test code that computes deltas from anchor boxes."""

    with tf.compat.v1.Session().as_default():
        pred_boxes = RegressBoxes.apply_bbox_deltas(boxes, deltas)

        res = pred_boxes.eval()

        np.testing.assert_array_almost_equal(res, outcome)


def test_regress_boxes_layer():
    mean = np.array([0.5, 0.5, 0.5, 0.5])
    std = np.array([0.1, 0.1, 0.1, 0.1])
    layer = RegressBoxes.RegressBoxes(mean=mean, std=std)

    np.testing.assert_array_equal(layer.mean, mean)
    np.testing.assert_array_equal(layer.std, std)


def test_regress_boxes_layer_set_anchors():
    """
    Make sure RegressBoxes works when anchors are baked-in as weights.
    """

    with tf.compat.v1.Session().as_default():
        layer_baked = RegressBoxes.RegressBoxes(anchor_shape=boxes.shape)
        layer_baked.set_anchors(boxes)

        out_baked = layer_baked([deltas])

        np.testing.assert_array_almost_equal(out_baked.eval(), outcome)


def test_regress_boxes_layer_input_anchors():
    """
    Make sure RegressBoxes works when anchors is set at runtime
    as an input.
    """

    with tf.compat.v1.Session().as_default():
        layer = RegressBoxes.RegressBoxes()

        out = layer([boxes, deltas])

        np.testing.assert_array_almost_equal(out.eval(), outcome)


def test_regress_boxes_layer_compute_output_shape():

    layer = RegressBoxes.RegressBoxes()
    shape = layer.compute_output_shape([[10], [10]])

    assert shape == [10]

