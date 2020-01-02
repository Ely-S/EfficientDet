# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
import cv2
import numpy as np

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def inject_keras_modules(func):
    import keras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = keras.backend
        kwargs['layers'] = keras.layers
        kwargs['models'] = keras.models
        kwargs['utils'] = keras.utils
        return func(*args, **kwargs)

    return wrapper


def inject_tfkeras_modules(func):
    import tensorflow.keras as tfkeras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = tfkeras.backend
        kwargs['layers'] = tfkeras.layers
        kwargs['models'] = tfkeras.models
        kwargs['utils'] = tfkeras.utils
        return func(*args, **kwargs)

    return wrapper


def init_keras_custom_objects():
    import keras
    import efficientnet as model

    custom_objects = {
        'swish': inject_keras_modules(model.get_swish)(),
        'FixedDropout': inject_keras_modules(model.get_dropout)()
    }

    keras.utils.generic_utils.get_custom_objects().update(custom_objects)


def init_tfkeras_custom_objects():
    import tensorflow.keras as tfkeras
    import efficientnet as model

    custom_objects = {
        'swish': inject_tfkeras_modules(model.get_swish)(),
        'FixedDropout': inject_tfkeras_modules(model.get_dropout)()
    }

    tfkeras.utils.get_custom_objects().update(custom_objects)


def normalize_image(image):
    """
    Normalize image by subtracting and dividing by the pre-computed
    mean and std. dev. Operates on image in-place
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image[..., 0] -= mean[0]
    image[..., 1] -= mean[1]
    image[..., 2] -= mean[2]
    image[..., 0] /= std[0]
    image[..., 1] /= std[1]
    image[..., 2] /= std[2]


def resize_image(image, image_size):
    """Resize images to image_size."""
    image_height, image_width = image.shape[:2]

    # check if image needs to be resized
    if image_height == image_width == image_size:
        new_image = image.astype(np.float32) / 255.0
        return new_image, 0, 0, 0

    # scale the image dimensions to fit into the new size without distortion
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    # Resize the input image to fit
    image = cv2.resize(image, (resized_width, resized_height))

    offset_h = (image_size - resized_height) // 2
    offset_w = (image_size - resized_width) // 2

    # Paste the input image into the center of a grey image of target size
    new_image = np.ones((image_size, image_size, 3), dtype=np.float32) * 128.
    new_image[offset_h:offset_h + resized_height,
              offset_w:offset_w + resized_width] = image.astype(np.float32)

    new_image /= 255.0

    return new_image, scale, offset_h, offset_w


def preprocess_image(image, image_size):
    resized_image, scale, offset_h, offset_w = resize_image(image, image_size)

    normalize_image(resized_image)

    return resized_image, scale, offset_h, offset_w
