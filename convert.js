from model import efficientdet
import time
import os

import cv2
import numpy as np

from utils import preprocess_image
from utils.anchors import anchors_for_shape

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

phi = 0
weighted_bifpn = True

model_path = 'pascal_10_0.6209_1.3075_0.7892.h5'
image_path = 'test/000010.jpg'

image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi]

classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

num_classes = len(classes)
score_threshold = 0.5
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

model, prediction_model = efficientdet(phi=phi,
                                       weighted_bifpn=weighted_bifpn,
                                       num_classes=num_classes,
                                       score_threshold=score_threshold)

prediction_model.load_weights(model_path, by_name=True)
