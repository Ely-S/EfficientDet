import numpy as np
import tensorflow as tf

import ClipBoxes


def test_clip_boxes_layer_compute_output_shape():
    layer = ClipBoxes.ClipBoxes()
    shape = layer.compute_output_shape([[10], [10]])

    assert shape == [10]


def test_clip_boxes_layer_call():
    boxes = np.array(
        # Batch Size
        [
            # boxes
            [
                # 4 coords
                [-0.1, 0, 1.3, 1],
                [10, 0, 210, 300],
                [-100, -0.5, 0.6, 180],
            ]
        ],
        dtype=np.float32,
    )

    # a batch of 32 200x200 3-channel images
    img = np.ones((32, 200, 200, 3))

    outcome = np.array(
        [
            [
                [0, 0, 1.3, 1],
                [10, 0, 199, 199],
                [0, 0, 0.6, 180]
            ]
        ],
        dtype=np.float32
    )

    inputs = np.array([img, boxes])

    with tf.compat.v1.Session().as_default():

        layer = ClipBoxes.ClipBoxes()
        result = layer.call(inputs)
        array = result.eval()

    np.testing.assert_array_equal(array, outcome)
