from model import efficientdet
import os

import tensorflowjs as tfjs

# Disable cuda for this becuase it slows down startup
os.environ['CUDA_VISIBLE_DEVICES'] = ''

model_path = 'pascal_10_0.6209_1.3075_0.7892.h5'
tfjs_target_dir = "jsmodel"

phi = 0
weighted_bifpn = True

VOC_CLASSES = 20
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi]

num_classes = VOC_CLASSES
score_threshold = 0.5

model, prediction_model = efficientdet(phi=phi,
                                       weighted_bifpn=weighted_bifpn,
                                       num_classes=num_classes,
                                       score_threshold=score_threshold)

prediction_model.load_weights(model_path, by_name=True)


tfjs.converters.save_keras_model(prediction_model, tfjs_target_dir)
