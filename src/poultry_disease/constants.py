"""Fixed facts about the trained model (from models/model_config.json of run_20261029_191001)."""

from __future__ import annotations

import json
from importlib import resources

#: Model output order. The packaged copy is checked against models/class_names.json by the tests.
CLASS_NAMES: tuple[str, ...] = tuple(
    json.loads(resources.files("poultry_disease.data").joinpath("class_names.json").read_text("utf-8"))
)
NUM_CLASSES = len(CLASS_NAMES)

CLASS_DISPLAY_NAMES = {
    "Coccidiosis": "Coccidiosis",
    "Healthy": "Healthy",
    "Salmonella": "Salmonella",
    "NCD": "Newcastle Disease (NCD)",
}

#: (height, width) of the model input. Aspect ratio is not preserved.
IMAGE_SIZE = (224, 224)

#: Layer names of the saved Keras model.
BACKBONE_LAYER = "resnet50_backbone"
GAP_LAYER = "global_average_pooling"
CLASSIFIER_LAYER = "classifier"
LAST_CONV_LAYER = "conv5_block3_out"

#: Spatial size of the final ResNet50 feature map (and of the CAM) for a 224 x 224 input.
CAM_SIZE = 7

#: resnet50.preprocess_input ("caffe" mode): RGB -> BGR, then subtract these per-channel means (BGR order).
CAFFE_MEANS_BGR = (103.939, 116.779, 123.68)

#: GitHub Release that hosts the weights (see models/checksums.sha256).
RELEASE_TAG = "v1.0.0"
RELEASE_BASE_URL = f"https://github.com/peerzadarabet1807/poultry-disease-detection/releases/download/{RELEASE_TAG}"
WEIGHTS_FILE = "poultry_resnet50.keras"
WEIGHTS_SHA256 = "7f05c52fdc7426de07d5d0ffb98a12b3157e87695fa86c68f32e611f0c9d665a"
