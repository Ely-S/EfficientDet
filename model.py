from functools import reduce

# from keras import layers
# from keras import initializers
# from keras import models
# from keras_ import EfficientNetB0, EfficientNetB1, EfficientNetB2
# from keras_ import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from layers import (
    ClipBoxes,
    RegressBoxes,
    FilterDetections,
    wBiFPNAdd,
    BatchNormalization,
)
from initializers import PriorProbability

from tensorflow.python.keras.utils import tf_utils
import tensorflow.keras as keras


w_bifpns = [64, 88, 112, 160, 224, 288, 384]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
]


def DepthwiseConvBlock(kernel_size, strides, name, freeze_bn=False):
    f1 = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name="{}_dconv".format(name),
    )
    f2 = BatchNormalization(freeze=freeze_bn, name="{}_bn".format(name))
    f3 = layers.ReLU(name="{}_relu".format(name))
    return reduce(
        lambda f, g: lambda *args, **kwargs: g(
            f(*args, **kwargs)), (f1, f2, f3)
    )


def ConvBlock(num_channels, name, kernel_size=1, strides=1, freeze_bn=False):
    f1 = layers.Conv2D(
        num_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name="{}_conv".format(name),
    )
    f2 = BatchNormalization(freeze=freeze_bn, name="{}_bn".format(name))
    f3 = layers.ReLU(name="{}_relu".format(name))
    return reduce(
        lambda f, g: lambda *args, **kwargs: g(
            f(*args, **kwargs)), (f1, f2, f3)
    )


def build_BiFPN(features, num_channels, layer_index, freeze_bn=False):
    if layer_index == 0:
        _, _, C3, C4, C5 = features
        P3_in = ConvBlock(
            num_channels,
            freeze_bn=freeze_bn,
            name="BiFPN_{}_P3".format(layer_index),
        )(C3)
        P4_in = ConvBlock(
            num_channels,
            freeze_bn=freeze_bn,
            name="BiFPN_{}_P4".format(layer_index),
        )(C4)
        P5_in = ConvBlock(
            num_channels,
            freeze_bn=freeze_bn,
            name="BiFPN_{}_P5".format(layer_index),
        )(C5)
        P6_in = ConvBlock(
            num_channels,
            kernel_size=3,
            strides=2,
            freeze_bn=freeze_bn,
            name="BiFPN_{}_P6".format(layer_index),
        )(C5)
        P7_in = ConvBlock(
            num_channels,
            kernel_size=3,
            strides=2,
            freeze_bn=freeze_bn,
            name="BiFPN_{}_P7".format(layer_index),
        )(P6_in)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P3_in = ConvBlock(
            num_channels,
            freeze_bn=freeze_bn,
            name="BiFPN_{}_P3".format(layer_index),
        )(P3_in)
        P4_in = ConvBlock(
            num_channels,
            freeze_bn=freeze_bn,
            name="BiFPN_{}_P4".format(layer_index),
        )(P4_in)
        P5_in = ConvBlock(
            num_channels,
            freeze_bn=freeze_bn,
            name="BiFPN_{}_P5".format(layer_index),
        )(P5_in)
        P6_in = ConvBlock(
            num_channels,
            freeze_bn=freeze_bn,
            name="BiFPN_{}_P6".format(layer_index),
        )(P6_in)
        P7_in = ConvBlock(
            num_channels,
            freeze_bn=freeze_bn,
            name="BiFPN_{}_P7".format(layer_index),
        )(P7_in)

    # upsample
    P7_U = layers.UpSampling2D()(P7_in)
    P6_td = layers.Add()([P7_U, P6_in])
    P6_td = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name="BiFPN_{}_U_P6".format(layer_index)
    )(P6_td)
    P6_U = layers.UpSampling2D()(P6_td)
    P5_td = layers.Add()([P6_U, P5_in])
    P5_td = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name="BiFPN_{}_U_P5".format(layer_index)
    )(P5_td)
    P5_U = layers.UpSampling2D()(P5_td)
    P4_td = layers.Add()([P5_U, P4_in])
    P4_td = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name="BiFPN_{}_U_P4".format(layer_index)
    )(P4_td)
    P4_U = layers.UpSampling2D()(P4_td)
    P3_out = layers.Add()([P4_U, P3_in])
    P3_out = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name="BiFPN_{}_U_P3".format(layer_index)
    )(P3_out)
    # downsample
    P3_D = layers.MaxPooling2D(strides=(2, 2))(P3_out)
    P4_out = layers.Add()([P3_D, P4_td, P4_in])
    P4_out = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name="BiFPN_{}_D_P4".format(layer_index)
    )(P4_out)
    P4_D = layers.MaxPooling2D(strides=(2, 2))(P4_out)
    P5_out = layers.Add()([P4_D, P5_td, P5_in])
    P5_out = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name="BiFPN_{}_D_P5".format(layer_index)
    )(P5_out)
    P5_D = layers.MaxPooling2D(strides=(2, 2))(P5_out)
    P6_out = layers.Add()([P5_D, P6_td, P6_in])
    P6_out = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name="BiFPN_{}_D_P6".format(layer_index)
    )(P6_out)
    P6_D = layers.MaxPooling2D(strides=(2, 2))(P6_out)
    P7_out = layers.Add()([P6_D, P7_in])
    P7_out = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name="BiFPN_{}_D_P7".format(layer_index)
    )(P7_out)

    return P3_out, P4_out, P5_out, P6_out, P7_out


def build_wBiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            C3)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            C4)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            C5)
        P6_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
            C5)
        P7_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
            P6_in)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            P3_in)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            P4_in)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            P5_in)
        P6_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
            P6_in)
        P7_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
            P7_in)

    # upsample
    P7_U = layers.UpSampling2D()(P7_in)
    P6_td = wBiFPNAdd(name='w_bi_fpn_add' if id ==
                      0 else 'w_bi_fpn_add_{}'.format(8 * id))([P7_U, P6_in])
    P6_td = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P6'.format(id))(P6_td)
    P6_U = layers.UpSampling2D()(P6_td)
    P5_td = wBiFPNAdd(name='w_bi_fpn_add_{}'.format(8 * id + 1))([P6_U, P5_in])
    P5_td = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P5'.format(id))(P5_td)
    P5_U = layers.UpSampling2D()(P5_td)
    P4_td = wBiFPNAdd(name='w_bi_fpn_add_{}'.format(8 * id + 2))([P5_U, P4_in])
    P4_td = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P4'.format(id))(P4_td)
    P4_U = layers.UpSampling2D()(P4_td)
    P3_out = wBiFPNAdd(name='w_bi_fpn_add_{}'.format(
        8 * id + 3))([P4_U, P3_in])
    P3_out = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P3'.format(id))(P3_out)
    # downsample
    P3_D = layers.MaxPooling2D(strides=(2, 2))(P3_out)
    P4_out = wBiFPNAdd(name='w_bi_fpn_add_{}'.format(
        8 * id + 4))([P3_D, P4_td, P4_in])
    P4_out = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P4'.format(id))(P4_out)
    P4_D = layers.MaxPooling2D(strides=(2, 2))(P4_out)
    P5_out = wBiFPNAdd(name='w_bi_fpn_add_{}'.format(
        8 * id + 5))([P4_D, P5_td, P5_in])
    P5_out = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P5'.format(id))(P5_out)
    P5_D = layers.MaxPooling2D(strides=(2, 2))(P5_out)
    P6_out = wBiFPNAdd(name='w_bi_fpn_add_{}'.format(
        8 * id + 6))([P5_D, P6_td, P6_in])
    P6_out = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P6'.format(id))(P6_out)
    P6_D = layers.MaxPooling2D(strides=(2, 2))(P6_out)
    P7_out = wBiFPNAdd(name='w_bi_fpn_add_{}'.format(
        8 * id + 7))([P6_D, P7_in])
    P7_out = DepthwiseConvBlock(
        kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P7'.format(id))(P7_out)

    return P3_out, P4_out, P5_out, P6_out, P7_out


def build_regress_head(width, depth, num_anchors=9):
    """
    Build a box prediction subnet as described in
    the RetinaNet Paper on page 5 https://arxiv.org/pdf/1708.02002.pdf
    """

    options = {
        "kernel_size": 3,
        "strides": 1,
        "padding": "same",
        "kernel_initializer": initializers.RandomNormal(
            mean=0.0, stddev=0.01, seed=None
        ),
        "bias_initializer": "zeros",
    }

    model = keras.Sequential(name="box_head")

    inputs = layers.Input(shape=(None, None, width))
    model.add(inputs)

    for i in range(depth):
        # @TODO: Switch all convs to depthwise sperable convs
        # when applicable
        conv = layers.Conv2D(
            filters=width,
            activation="relu",
            name="regress_head_conv_{}".format(i),
            **options)

        model.add(conv)

    # Note: there is no activation function at the end. The offsets are linear.
    final_conv = layers.Conv2D(
        filters=num_anchors * 4,
        name="regress_head_conv_final",
        **options)

    model.add(final_conv)

    # output (batch_size, num_anchors_this_feature_map, 4)
    model.add(layers.Reshape((-1, 4)))

    return model

    # inputs = layers.Input(shape=(None, None, width))
    # outputs = inputs

    # # @TODO: Switch all convs to depthwise sperable convs with swish activation
    # # When applicable. The last layer of regression and classification heads needs
    # # To have a relu activation because the minimum must be 0.
    # for i in range(depth):
    #     outputs = layers.Conv2D(
    #         filters=width,
    #         activation="relu",
    #         name="regress_head_conv_{}".format(i),
    #         ** options)(outputs)

    # # Note: there is no activation function at the end. The offsets are linear.
    # outputs = layers.Conv2D(num_anchors * 4,
    #                         name="regress_head_conv_final",
    #                         ** options)(outputs)
    # # (b, num_anchors_this_feature_map, 4)
    # outputs = layers.Reshape((-1, 4))(outputs)

    # return models.Model(inputs=inputs, outputs=outputs, name="box_head")


def build_class_head(width, depth, num_classes=20, num_anchors=9):
    options = {
        "kernel_size": 3,
        "strides": 1,
        "padding": "same",
    }

    inputs = layers.Input(shape=(None, None, width))
    outputs = inputs

    for i in range(depth):
        outputs = layers.Conv2D(
            filters=width,
            activation="relu",

            # See: https://arxiv.org/pdf/1708.02002.pdf page 5 under initialization
            kernel_initializer=initializers.RandomNormal(
                mean=0.0, stddev=0.01, seed=None
            ),

            name="class_head_{}".format(i),
            bias_initializer="zeros",
            **options,
        )(outputs)

    # outputs = layers.Conv2D(num_anchors * num_classes, **options)(outputs)
    outputs = layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=initializers.RandomNormal(
            mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=0.01),
        name="pyramid_classification",
        **options,
    )(outputs)

    # (b, num_anchors_this_feature_map, 4)
    outputs = layers.Reshape((-1, num_classes))(outputs)
    outputs = layers.Activation('sigmoid')(outputs)

    return models.Model(inputs=inputs, outputs=outputs, name="class_head")


# class BoxHead(tf.layers.Layer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def __call__(self, input):
#         def build_regress_head(width, depth, num_anchors=9):

#     def build(self, input_shape):
#         options = {
#             "kernel_size": 3,
#             "strides": 1,
#             "padding": "same",
#             "kernel_initializer": initializers.RandomNormal(
#                 mean=0.0, stddev=0.01, seed=None
#             ),
#             "bias_initializer": "zeros",
#         }

#     self.layers = []

#     for i in range(depth):
#         # @TODO: Switch all convs to depthwise sperable convs
#         # when applicable
#         conv = layers.Conv2D(
#             filters=width,
#             activation="relu",
#             name="regress_head_conv_{}".format(i),
#             **options)

#         self.layers.append(conv)

#     # Note: there is no activation function at the end. The offsets are linear.
#     final_conv = layers.Conv2D(
#         filters=num_anchors * 4,
#         name="regress_head_conv_final",
#         **options)

#     self.layers.append(final_conv)

#     # output (batch_size, num_anchors_this_feature_map, 4)
#     self.layers.append(layers.Reshape((-1, 4)))


def efficientdet(
    phi,
    num_classes=20,
    weighted_bifpn=False,
    freeze_bn=False,
    score_threshold=0.01,
    no_filter=False,
    anchors=None,
    just_training_model=False,
    **bbkwargs
):
    assert phi in range(7)
    input_size = image_sizes[phi]
    input_shape = (input_size, input_size, 3)

    image_input = layers.Input(input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = 2 + phi
    w_head = w_bifpn
    box_head_depth = 3 + int(phi / 3)
    backbone_cls = backbones[phi]

    # features = backbone_cls(include_top=False, input_shape=input_shape, weights=weights)(image_input)
    features = backbone_cls(input_tensor=image_input,
                            freeze_bn=freeze_bn, **bbkwargs)

    if weighted_bifpn:
        for i in range(d_bifpn):
            features = build_wBiFPN(features, w_bifpn, i, freeze_bn=freeze_bn)
    else:
        for i in range(d_bifpn):
            features = build_BiFPN(features, w_bifpn, i, freeze_bn=freeze_bn)

    regress_head = build_regress_head(w_head, box_head_depth)
    class_head = build_class_head(
        w_head, box_head_depth, num_classes=num_classes)

    regression = [regress_head(feature) for feature in features]
    regression = layers.Concatenate(axis=1, name="regression")(regression)

    classification = [class_head(feature) for feature in features]
    classification = layers.Concatenate(
        axis=1, name="classification")(classification)

    model_inputs = [image_input]

    model = models.Model(
        inputs=model_inputs,
        outputs=[regression, classification],
        name="efficientdet"
    )

    if just_training_model:
        return model

    # Optionally allow anchors to be baked into the model.
    # this is useful for exporting a single package to js
    if anchors is not None:
        anchor_array = np.expand_dims(anchors, axis=0)

        box_predicter = RegressBoxes(
            name="boxes", anchor_shape=anchor_array.shape)
        box_predicter.set_anchors(anchor_array)
        boxes = box_predicter([regression])

    else:
        # apply predicted regression to anchors
        anchors_input = layers.Input((None, 4))
        boxes = RegressBoxes(name="boxes")([anchors_input, regression])

        model_inputs.append(anchors_input)

    boxes = ClipBoxes(name="clipped_boxes")([image_input, boxes])

    # Don't implement filter detections as layer so it can be implemented as a basic js function
    # filter detections (apply NMS / score threshold / select top-k)
    if no_filter:
        prediction_model = models.Model(
            inputs=model_inputs,
            outputs=[boxes, classification],
            name="efficientdet_p",
        )

        return model, prediction_model

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        name="filtered_detections",
        score_threshold=score_threshold
    )([boxes, classification])

    prediction_model = models.Model(
        inputs=model_inputs, outputs=detections, name="efficientdet_p"
    )

    return model, prediction_model
