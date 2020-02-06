import numpy as np
import tensorflow as tf

import FilterDetections as fd

# dimension of this array is
# (batch size, num boxes, 4)
boxes = np.array(
        # boxes
    [
            # 4 coords
            [0, 0, 1, 1],
            [0.5, 0.5, 0.6, 0.6],
            [0.1, 0.1, 0.6, 0.6],
    ], "float32"
)
scores = np.array([.6, .2, .1], "float32")
labels = np.array([1, 2, 1], "int64")


def test_filter_by_score_and_nms():
    with tf.compat.v1.Session().as_default():
        detections = fd.filter_by_score_and_nms(
            scores, labels, .12, boxes, 3, .5
        )

        np.testing.assert_array_almost_equal(
            detections.eval(), np.array([[0, 1], [1, 2]]))
