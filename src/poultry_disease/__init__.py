"""Image-based classification of poultry fecal photographs (research prototype, not a veterinary diagnosis).

The package reproduces the training notebook's preprocessing and inference exactly, adds class activation
maps (CAM, equal to the notebook's Grad-CAM for this architecture) and fetches the released weights.

    from poultry_disease import Predictor
    predictor = Predictor.from_release()
    predictor.predict_image("photo.jpg")
"""

from poultry_disease.constants import CLASS_NAMES, IMAGE_SIZE
from poultry_disease.predict import Prediction, Predictor

__version__ = "1.0.0"
__all__ = ["CLASS_NAMES", "IMAGE_SIZE", "Prediction", "Predictor", "__version__"]
