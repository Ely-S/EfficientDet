import argparse
import time
import os

from model import efficientdet
from utils import preprocess_image
from utils.anchors import anchors_for_shape

import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument(
    "--phi", type=int, required=True, help="phi constant between 0 and 7 inclusive"
)

parser.add_argument(
    "--cuda-visible-devices", default="", help="Set the GPU to use or default  to CPU"
)

parser.add_argument(
    "--not-weighted",
    default=False,
    help="Run with an unweighted biFPN model." "Defaults to use weighted BiFPN",
)

parser.add_argument("--model", required=True, help="path to .h5 file")

parser.add_argument("image")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

weighted_bifpn = not args.not_weighted

classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

num_classes = len(classes)

score_threshold = 0.5
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[args.phi]

anchors = anchors_for_shape((image_size, image_size))

model, prediction_model = efficientdet(
    phi=args.phi,
    weighted_bifpn=weighted_bifpn,
    num_classes=num_classes,
    score_threshold=score_threshold,
    anchors=anchors
)

prediction_model.load_weights(args.model, by_name=True)


image = cv2.imread(args.image)

if image is None:
    raise FileNotFoundError(args.image)

src_image = image.copy()
image = image[:, :, ::-1]
h, w = image.shape[:2]

image, scale, offset_h, offset_w = preprocess_image(image, image_size=image_size)

inputs = np.expand_dims(image, axis=0)

# run network
start = time.time()
boxes_list, scores, labels = prediction_model.predict_on_batch(
    [np.expand_dims(image, axis=0)]
)


boxes = boxes_list[0]

print("scale", scale, "offset_h", offset_h, "offset_w", offset_w)
print("boxes", boxes.shape, boxes, sep="\n")
print("scores", scores.shape, scores, sep="\n")
print("labels", labels.shape, labels, sep="\n")

print(time.time() - start)

# Subtract offset_w from every predicted x1 and x2
boxes[:, [0, 2]] = boxes[:, [0, 2]] - offset_w
# Subtract offset_w from every predicted y1 and y2
boxes[:, [1, 3]] = boxes[:, [1, 3]] - offset_h

# Divide boxes by scale
boxes /= scale

# ensure that predicted x1 and x2 coordinates between 0 and width-1
boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)

# ensure that predicted y1 and y2 coordinates between 0 and height-1
boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)

# select indices which have a score above the threshold
indices = np.where(scores[0, :] > score_threshold)[0]

print(indices)
# select those detections
detections = boxes[indices]
scores = scores[0, indices]
labels = labels[0, indices]


print("final")
print(boxes)
print(scores)
print(labels)

for box, score, label in zip(detections, scores, labels):
    xmin = int(round(box[0]))
    ymin = int(round(box[1]))
    xmax = int(round(box[2]))
    ymax = int(round(box[3]))
    score = "{:.4f}".format(score)
    class_id = int(label)
    color = colors[class_id]
    class_name = classes[class_id]
    label = "-".join([class_name, score])

    ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
    cv2.rectangle(
        src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1
    )

    cv2.putText(
        src_image,
        label,
        (xmin, ymax - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", src_image)
cv2.waitKey(0)

print("done")
