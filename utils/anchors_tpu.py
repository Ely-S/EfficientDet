
import numpy as np
import utils.anchors


def gen_anchor_targets(
    anchors,
    image,
    bboxes,
    labels,
    num_classes,
    negative_overlap=0.4,
    positive_overlap=0.5
):
    """ Generate anchor targets for bbox detection.

    @author: Eli

    This is a version of anchor_targets_bbox that takes tensors for images, bboxes, and labels
    to play nice with tensorflow.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of images.
        bboxes_group: np.array(n, x1, y1, x2, y2)
        labels_grpup: np.array(n)

        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_target: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_target: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
                      where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
                      last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    regression_target = np.zeros(
        (anchors.shape[0], 4 + 1), dtype=np.float32)

    labels_target = np.zeros(
        (anchors.shape[0], num_classes + 1), dtype=np.float32)

    # compute labels and regression targets
    if bboxes.shape[0]:
        # obtain indices of ground truth annotations with the greatest overlap
        positive_indices, ignore_indices, argmax_overlaps_inds = utils.anchors.compute_gt_annotations(
            anchors, bboxes, negative_overlap, positive_overlap)

        labels_target[ignore_indices, -1] = -1
        labels_target[positive_indices, -1] = 1

        regression_target[ignore_indices, -1] = -1
        regression_target[positive_indices, -1] = 1

        # compute target class labels
        labels_target[positive_indices, labels
                      [argmax_overlaps_inds[positive_indices]].astype(int)] = 1

        regression_target[:, : -1] = utils.anchors.bbox_transform(
            anchors, bboxes[argmax_overlaps_inds, :])

    # ignore annotations outside of image
    anchors_centers = np.vstack(
        [(anchors[:, 0] + anchors[:, 2]) / 2,
            (anchors[:, 1] + anchors[:, 3]) / 2]).T

    outside_indices = np.logical_or(
        anchors_centers[:, 0] >= image.shape[1],
        anchors_centers[:, 1] >= image.shape[0])

    # -1 means ignore
    labels_target[outside_indices, -1] = -1
    regression_target[outside_indices, -1] = -1

    return regression_target, labels_target
