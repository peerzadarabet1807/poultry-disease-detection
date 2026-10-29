"""Load the trained Keras model and attach an in-graph class activation map (CAM) output.

The trained head is ``GlobalAveragePooling -> Dropout -> Dense(4, softmax)`` on top of the ResNet50 feature
map ``A`` (``conv5_block3_out``, 7 x 7 x 2048). The notebook's Grad-CAM takes the gradient of the class logit
``y^c = sum_k w_k^c mean_ij(A^k_ij) + b^c`` with respect to ``A``, which is ``w_k^c / (H W)`` at every
position. Grad-CAM is therefore ``ReLU(sum_k w_k^c A^k) / (H W)``: classic CAM up to a positive constant
that disappears when the map is normalised. ``sum_k w_k^c A^k`` is a 1 x 1 convolution of ``A`` with the
Dense kernel, so one forward pass can return the probabilities and all four class maps.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402

from poultry_disease.constants import (  # noqa: E402
    BACKBONE_LAYER,
    CLASSIFIER_LAYER,
    GAP_LAYER,
    IMAGE_SIZE,
    NUM_CLASSES,
)

INPUT_NAME = "image"
PROBS_OUTPUT = "probs"
CAM_OUTPUT = "cam"


def load_keras_model(path: str | Path) -> keras.Model:
    """The trained model as saved by the notebook (input: preprocessed 224 x 224 x 3, output: softmax)."""
    return keras.models.load_model(str(path), compile=False)


def build_inference_model(trained: keras.Model) -> keras.Model:
    """Trained model + CAM head. Outputs ``probs`` [N, 4] and ``cam`` [N, 7, 7, 4] (pre-ReLU class maps).

    The trained layer objects are reused, so the probabilities are computed by exactly the same weights.
    Only the 1 x 1 ``cam`` convolution is new, and its kernel is a reshaped copy of the Dense kernel.
    """
    backbone = trained.get_layer(BACKBONE_LAYER)
    gap = trained.get_layer(GAP_LAYER)
    classifier = trained.get_layer(CLASSIFIER_LAYER)

    image = keras.Input((*IMAGE_SIZE, 3), dtype="float32", name=INPUT_NAME)
    fmap = backbone(image, training=False)
    probs = keras.layers.Identity(name=PROBS_OUTPUT)(classifier(gap(fmap)))

    cam_layer = keras.layers.Conv2D(NUM_CLASSES, 1, use_bias=False, name=CAM_OUTPUT)
    cam = cam_layer(fmap)
    kernel = classifier.get_weights()[0]  # (2048, NUM_CLASSES)
    cam_layer.set_weights([kernel.reshape(1, 1, *kernel.shape)])

    return keras.Model(image, {PROBS_OUTPUT: probs, CAM_OUTPUT: cam}, name="poultry_resnet50_cam")
