#!/usr/bin/env python3
"""
hazard_detection_api.py

A FastAPI service that loads a YOLOv8 object-detection model (Ultralytics)
trained on road-surface hazard imagery and exposes:

    POST /verify-hazard-image
        Accepts an uploaded field photo (multipart/form-data, e.g. from a
        mobile app's camera/gallery picker). Runs YOLOv8 inference, returns
        the top detected hazard class + confidence, the bounding box, and
        an annotated copy of the image with the box drawn on it.

    GET /health
        Liveness + "is the model loaded, what classes does it know" check.

Target classes (see train_hazard_yolov8.py to train the model):
    road_crack, active_mudslide, blocked_road, normal

Run it with:
    uvicorn hazard_detection_api:app --reload --port 8001

Environment variables:
    HAZARD_MODEL_PATH   Path to the trained .pt weights
                         (default: hazard_yolov8.pt)
    HAZARD_CONF_DEFAULT Default confidence threshold (default: 0.25)
    HAZARD_MAX_UPLOAD_MB Max accepted upload size in MB (default: 15)
    HAZARD_MAX_DIM      Max image dimension inference/annotation is run at,
                         larger photos are downscaled first (default: 1280)

--------------------------------------------------------------------------
Mobile-upload specifics handled here
--------------------------------------------------------------------------
* EXIF orientation: phone photos are frequently stored "sideways" with an
  EXIF orientation tag rather than physically rotated pixels. Images are
  opened with Pillow and auto-rotated (`ImageOps.exif_transpose`) before
  inference, or the model would see (and box) a rotated scene.
* HEIC/HEIF: iPhones default to HEIC, which OpenCV and stock Pillow can't
  decode. If the optional `pillow-heif` package is installed, its opener
  is registered automatically and HEIC uploads just work; otherwise those
  uploads fail with a clear 415 error telling the caller to convert to
  JPEG/PNG client-side or install pillow-heif server-side.
* Large photos: a modern phone photo can be 12+ MP. It's downscaled to
  HAZARD_MAX_DIM on the long edge before inference and before it's
  returned as the annotated image, to keep latency and response size
  reasonable -- detection quality is essentially unaffected since YOLOv8
  internally resizes to its training imgsz anyway.
"""

from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

# Ultralytics will, by default, silently `pip install` optional dependencies
# (e.g. a HEIC-decoding helper) the first time it thinks it needs one -- a
# live network/pip call triggered by whatever bytes a caller happens to
# upload. That's not something a request-handling service should be doing
# on its own; set this BEFORE importing ultralytics (done lazily in
# load_model()) so it never attempts it. Add packages to requirements.txt
# and install them deliberately instead.
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "True")

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HAS_HEIF = True
except ImportError:
    _HAS_HEIF = False


MODEL_PATH = os.environ.get("HAZARD_MODEL_PATH", "hazard_yolov8.pt")
DEFAULT_CONF = float(os.environ.get("HAZARD_CONF_DEFAULT", "0.25"))
MAX_UPLOAD_MB = float(os.environ.get("HAZARD_MAX_UPLOAD_MB", "15"))
MAX_DIM = int(os.environ.get("HAZARD_MAX_DIM", "1280"))

EXPECTED_CLASSES = {"road_crack", "active_mudslide", "blocked_road", "normal"}
BOX_COLOR_PRIMARY = (0, 0, 255)     # red, BGR -- the top/primary detection
BOX_COLOR_SECONDARY = (0, 165, 255)  # orange, BGR -- other above-threshold detections
NORMAL_COLOR = (0, 200, 0)          # green, BGR -- "verified clear" overlay


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
class ModelState:
    model = None
    class_names: dict[int, str] = {}


model_state = ModelState()


def load_model(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model weights not found at '{path}'. Train one with "
            f"train_hazard_yolov8.py, or set HAZARD_MODEL_PATH."
        )
    from ultralytics import YOLO  # imported lazily so --help-style tooling stays light

    model_state.model = YOLO(path)
    model_state.class_names = model_state.model.names

    names_set = set(model_state.class_names.values())
    if names_set != EXPECTED_CLASSES:
        print(
            f"[WARN] Loaded model's classes {sorted(names_set)} don't match the "
            f"expected hazard classes {sorted(EXPECTED_CLASSES)}. Predictions will "
            f"still be served, but class names in responses will reflect whatever "
            f"this checkpoint was actually trained on."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(MODEL_PATH)
    yield


app = FastAPI(
    title="Road Hazard Verification API",
    description="Detects road-surface hazards (cracks, active mudslides, "
                 "blocked roads) in field photos using YOLOv8.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
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
    filename: Optional[str]
    image_width: int
    image_height: int
    hazard_class: str
    confidence: float = Field(..., description="Confidence of the reported hazard_class, 0.0-1.0.")
    hazard_detected: bool = Field(..., description="False when the top result is 'normal' (no hazard).")
    bounding_box: Optional[BoundingBox] = Field(
        None, description="Box for the primary detection; null when nothing cleared the confidence threshold."
    )
    all_detections: list[DetectionWithBox] = Field(
        ..., description="Every detection above the confidence threshold, most confident first."
    )
    annotated_image: str = Field(
        ..., description="data:image/jpeg;base64,... JPEG of the (possibly downscaled) photo with boxes drawn."
    )
    model_classes: list[str]
    confidence_threshold: float
    inference_ms: float


# --------------------------------------------------------------------------- #
# Image decoding (EXIF-aware, HEIC-aware, size-capped)
# --------------------------------------------------------------------------- #
def decode_upload_to_bgr(raw_bytes: bytes, content_type: Optional[str]) -> np.ndarray:
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)  # correct phone-camera rotation
        pil_img = pil_img.convert("RGB")
    except Exception as exc:
        hint = ""
        if content_type and "heic" in content_type.lower() and not _HAS_HEIF:
            hint = " (HEIC upload detected but pillow-heif isn't installed server-side; " \
                   "install pillow-heif, or convert to JPEG/PNG before uploading.)"
        raise HTTPException(
            status_code=415,
            detail=f"Could not decode the uploaded file as an image.{hint} ({exc})",
        )

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest > MAX_DIM:
        scale = MAX_DIM / longest
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return bgr


# --------------------------------------------------------------------------- #
# Inference + drawing
# --------------------------------------------------------------------------- #
def run_inference(img_bgr: np.ndarray, conf: float, iou: float) -> list[dict]:
    results = model_state.model.predict(source=img_bgr, conf=conf, iou=iou, verbose=False)
    result = results[0]

    h, w = img_bgr.shape[:2]
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "hazard_class": model_state.class_names.get(cls_id, str(cls_id)),
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "bbox_norm": (x1 / w, y1 / h, x2 / w, y2 / h),
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def draw_annotations(img_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = img_bgr.copy()

    if not detections:
        cv2.putText(
            annotated, "No hazard detected", (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, NORMAL_COLOR, 2, cv2.LINE_AA,
        )
        return annotated

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        is_primary = i == 0
        color = BOX_COLOR_PRIMARY if is_primary else BOX_COLOR_SECONDARY
        thickness = 3 if is_primary else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{det['hazard_class']} {det['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 8, th + 4)
        cv2.rectangle(annotated, (x1, label_y - th - 6), (x1 + tw + 6, label_y + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 3, label_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def encode_jpeg_data_uri(img_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode annotated image as JPEG.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if model_state.model is not None else "model_not_loaded",
        "model_classes": list(model_state.class_names.values()),
        "heic_support": _HAS_HEIF,
    }


@app.post("/verify-hazard-image", response_model=HazardVerificationResponse)
async def verify_hazard_image(
    file: UploadFile = File(..., description="Field photo (JPEG/PNG/HEIC if pillow-heif is installed)."),
    confidence: float = Query(DEFAULT_CONF, ge=0.0, le=1.0, description="Minimum confidence to report a detection."),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold for non-max suppression."),
) -> HazardVerificationResponse:
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"Expected an image upload, got '{file.content_type}'.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Upload exceeds the {MAX_UPLOAD_MB} MB limit.")

    img_bgr = decode_upload_to_bgr(raw_bytes, file.content_type)
    h, w = img_bgr.shape[:2]

    start = time.perf_counter()
    try:
        detections = run_inference(img_bgr, conf=confidence, iou=iou)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    inference_ms = (time.perf_counter() - start) * 1000

    annotated = draw_annotations(img_bgr, detections)
    annotated_data_uri = encode_jpeg_data_uri(annotated)

    if detections:
        top = detections[0]
        hazard_class = top["hazard_class"]
        top_confidence = top["confidence"]
        bx1, by1, bx2, by2 = top["bbox"]
        nx1, ny1, nx2, ny2 = top["bbox_norm"]
        bounding_box = BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2,
                                    x1_norm=round(nx1, 4), y1_norm=round(ny1, 4),
                                    x2_norm=round(nx2, 4), y2_norm=round(ny2, 4))
    else:
        hazard_class = "normal"
        top_confidence = 1.0  # nothing cleared the confidence threshold -> treat as verified clear
        bounding_box = None

    all_detections = [
        DetectionWithBox(
            hazard_class=d["hazard_class"],
            confidence=round(d["confidence"], 4),
            bounding_box=BoundingBox(
                x1=d["bbox"][0], y1=d["bbox"][1], x2=d["bbox"][2], y2=d["bbox"][3],
                x1_norm=round(d["bbox_norm"][0], 4), y1_norm=round(d["bbox_norm"][1], 4),
                x2_norm=round(d["bbox_norm"][2], 4), y2_norm=round(d["bbox_norm"][3], 4),
            ),
        )
        for d in detections
    ]

    return HazardVerificationResponse(
        filename=file.filename,
        image_width=w,
        image_height=h,
        hazard_class=hazard_class,
        confidence=round(top_confidence, 4),
        hazard_detected=hazard_class != "normal",
        bounding_box=bounding_box,
        all_detections=all_detections,
        annotated_image=annotated_data_uri,
        model_classes=list(model_state.class_names.values()),
        confidence_threshold=confidence,
        inference_ms=round(inference_ms, 1),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("hazard_detection_api:app", host="0.0.0.0", port=8001, reload=True)
