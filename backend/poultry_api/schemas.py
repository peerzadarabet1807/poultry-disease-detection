"""Response models (they also document the API in /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Image-based classification prototype for research. Not a veterinary diagnosis; it does not replace a "
    "veterinarian or a laboratory test."
)


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str


class Verdict(BaseModel):
    category: Literal["disease_pattern", "no_disease_pattern", "uncertain"] = Field(
        description="'uncertain' when the top probability or the top-2 margin is below the usability thresholds."
    )
    top_class: str
    runner_up: str
    top_probability: float
    margin: float = Field(description="Top probability minus the second highest.")
    reasons: list[Literal["low_top_probability", "small_margin"]]
    notifiable: bool = Field(description="True when Newcastle disease (NCD) is the most likely class.")


class ImageInfo(BaseModel):
    format: Literal["jpeg", "png", "heic"]
    width: int
    height: int


class Cam(BaseModel):
    target_class: str = Field(description="Class the overlay explains (the predicted class).")
    maps: dict[str, list[list[float]]] = Field(
        description="Per class, the 7 x 7 class activation map after ReLU, divided by its maximum (0 to 1)."
    )
    overlay_png_base64: str = Field(description="224 x 224 PNG: jet heatmap at alpha 0.4 over the input.")
    input_png_base64: str = Field(description="224 x 224 PNG: the image exactly as the model saw it.")


class PredictResponse(BaseModel):
    model_version: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    low_confidence: bool
    verdict: Verdict
    image: ImageInfo
    cam: Cam | None = None
    disclaimer: str = DISCLAIMER


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
