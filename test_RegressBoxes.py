import numpy as np
import tensorflow as tf

import RegressBoxes


def test_apply_bbox_deltas():
    """Test code that computes deltas from anchor boxes."""
    # dimension of this array is
    # (batch size, num boxes, 4)
    boxes = np.array(
        # Batch Size
        [
            # boxes
            [
                # 4 coords
                [0, 0, 1, 1],
                [.5, .5, .6, .6]
            ]
        ],
    )

    # deltas is the same shape as boxes
    # and during operation contains the predicted
    # deltas from anchors
    deltas = np.array(
        # batch size
        [
            # boxes
            [
                # coords
                [.1, .1, .1, .1],
                [-.2, -.2, .2, .2]
            ]
        ]
    )

    outcome = np.array(
        [
            [
                [0.02,  0.02, 1.02,  1.02],
                [0.496, 0.496, 0.604, 0.604]
            ]
        ]
    )

    with tf.Session().as_default():
        pred_boxes = RegressBoxes.apply_bbox_deltas(
            boxes, deltas)

        res = pred_boxes.eval()

        np.testing.assert_array_almost_equal(
            res, outcome
        )


def test_regress_boxes_layer():
    mean = np.array([0.5, 0.5, 0.5, 0.5])
    std = np.array([0.1, 0.1, 0.1, 0.1])
    layer = RegressBoxes.RegressBoxes(mean=mean, std=std)

    np.testing.assert_array_equal(layer.mean, mean)
    np.testing.assert_array_equal(layer.std, std)


def test_regress_boxes_layer_compute_output_shape():

    layer = RegressBoxes.RegressBoxes()
    shape = layer.compute_output_shape([[10], [10]])
    
    assert shape == [10]

