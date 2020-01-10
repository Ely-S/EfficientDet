from model import efficientdet
import os

import tensorflowjs as tfjs
from kito import reduce_keras_model

import argparse

parser = argparse.ArgumentParser(description="Convert to tfjs")
parser.add_argument("h5file", type=str, help="path to h5 file")
parser.add_argument("out_dir", help="will create this directory if doesn't exist")
parser.add_argument("--summary", action="store_true", default=False)
parser.add_argument(
    "--kito", action="store_true", default=False, help="Reduce a model using kito",
)
parser.add_argument(
    "--weighted",
    action="store_true",
    default=False,
    help="use a model model with weighted biFPN layers. currently not supported",
)
parser.add_argument("--quantization_dtype", default=None)

args = parser.parse_args()


# Disable cuda for this becuase it slows down startup
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_path = args.h5file  #'pascal_10_0.6209_1.3075_0.7892.h5'
tfjs_target_dir = args.out_dir

phi = 1
weighted_bifpn = args.weighted

VOC_CLASSES = 20
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi]

num_classes = VOC_CLASSES
score_threshold = 0.5

model, prediction_model = efficientdet(
    phi=phi,
    weighted_bifpn=weighted_bifpn,
    num_classes=num_classes,
    score_threshold=score_threshold,
    drop_connect_rate=0,  # Remove dropout layers
)

prediction_model.load_weights(model_path, by_name=True)

if args.kito:
    print("running Keras Inference Time Optimization")
    final_model = reduce_keras_model(prediction_model)
else:
    final_model = prediction_model

if args.summary:
    final_model.summary()

tfjs.converters.save_keras_model(final_model, tfjs_target_dir, args.quantization_dtype)

