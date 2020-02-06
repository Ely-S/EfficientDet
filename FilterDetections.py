from tensorflow import keras
import tensorflow as tf


def filter_by_score_and_nms(
    scores_, labels_, score_threshold, boxes, max_detections, iou_threshold
):
    """Return indicies above a certain  threshold and apply NMS. """
    indices_ = tf.compat.v1.where(keras.backend.greater(scores_, score_threshold))

    if iou_threshold > 0:
        filtered_boxes = tf.gather_nd(boxes, indices_)
        filtered_scores = keras.backend.gather(scores_, indices_)[:, 0]

        # perform NMS
        nms_indices = tf.image.non_max_suppression(
            filtered_boxes,
            filtered_scores,
            max_output_size=max_detections,
            iou_threshold=iou_threshold,
        )

        # filter indices based on NMS
        # (num_score_nms_keeps, 1)
        indices_ = tf.gather(indices_, nms_indices)

    # add indices to list of all indices
    # (num_score_nms_keeps, )
    labels_ = tf.gather_nd(labels_, indices_)

    # (num_score_nms_keeps, 2)
    indices_ = tf.stack([indices_[:, 0], labels_], axis=1)

    return indices_


def filter_detections(
    boxes,
    classification,
    class_specific_filter=True,
    score_threshold=0.01,
    max_detections=300,
    iou_threshold=0.5,
    ):
    """
    Filter detections using the boxes and classification values.

    Args
        boxes: Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification: Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other: List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
        nms: Flag to enable/disable non maximum suppression.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        iou_threshold: Threshold for the IoU value to determine when a box should be suppressed. Set to 0 to disable Non Max Suppression.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
            other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((keras.backend.shape(scores)[0],), dtype="int64")
            all_indices.append(
                filter_by_score_and_nms(
                    scores,
                    labels,
                    score_threshold,
                    boxes,
                    max_detections,
                    iou_threshold,
                )
            )

        # concatenate indices to single tensor
        # (concatenated_num_score_nms_keeps, 2)
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = keras.backend.max(classification, axis=1)
        labels = keras.backend.argmax(classification, axis=1)
        indices = filter_by_score_and_nms(
            scores, labels, score_threshold, boxes, max_detections,
            iou_threshold
        )

    # select top k
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(
        scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0])
    )

    # filter input using the final set of indices
    indices = keras.backend.gather(indices[:, 0], top_indices)
    boxes = keras.backend.gather(boxes, indices)
    labels = keras.backend.gather(labels, top_indices)

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes = tf.pad(tensor=boxes, paddings=[[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(tensor=scores, paddings=[[0, pad_size]], constant_values=-1)
    labels = tf.pad(tensor=labels, paddings=[[0, pad_size]], constant_values=-1)
    labels = keras.backend.cast(labels, "int32")

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])

    return [boxes, scores, labels]


class FilterDetections(keras.layers.Layer):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
        nms=True,
        class_specific_filter=True,
        nms_threshold=0.5,
        score_threshold=0.01,
        max_detections=300,
        parallel_iterations=32,
        **kwargs
    ):
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms: Flag to enable/disable NMS.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            parallel_iterations: Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes_ = args[0]
            classification_ = args[1]

            if not self.nms:
                self.nms_threshold = 0

            return filter_detections(
                boxes_,
                classification_,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                iou_threshold=self.nms_threshold,
            )

        # call filter_detections on each batch item
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification],
            dtype=[keras.backend.floatx(), keras.backend.floatx(), "int32"],
            parallel_iterations=self.parallel_iterations,
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ]

    def compute_mask(self, inputs, mask=None):
        """
        This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update(
            {
                "nms": self.nms,
                "class_specific_filter": self.class_specific_filter,
                "nms_threshold": self.nms_threshold,
                "score_threshold": self.score_threshold,
                "max_detections": self.max_detections,
                "parallel_iterations": self.parallel_iterations,
            }
        )

        return config
