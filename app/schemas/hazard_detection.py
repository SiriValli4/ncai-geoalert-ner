"""Response contract for POST /hazards/verify-image.
Unchanged from legacy/hazard_detection_api.py so existing clients keep working.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    x1_norm: float
    y1_norm: float
    x2_norm: float
    y2_norm: float


class Detection(BaseModel):
    hazard_class: str
    confidence: float


class DetectionWithBox(Detection):
    bounding_box: BoundingBox


class HazardVerificationResponse(BaseModel):
    filename: str | None
    image_width: int
    image_height: int
    hazard_class: str
    confidence: float = Field(..., description="Confidence of hazard_class, 0.0-1.0.")
    hazard_detected: bool = Field(..., description="False when the top result is 'normal'.")
    bounding_box: BoundingBox | None = Field(
        None, description="Primary detection's box; null when nothing cleared the threshold."
    )
    all_detections: list[DetectionWithBox] = Field(
        ..., description="Every detection above the threshold, most confident first."
    )
    annotated_image: str = Field(
        ..., description="data:image/jpeg;base64,... of the photo with boxes drawn."
    )
    model_classes: list[str]
    confidence_threshold: float
    inference_ms: float
