"""YOLOv8 road-hazard detection: image decoding, inference, annotation.

Ported from legacy/hazard_detection_api.py. Weights are produced by
ml/training/train_hazard_yolov8.py. Heavy imports (cv2, PIL, ultralytics)
are lazy so the API boots without the `vision` extra installed; the
verify-image endpoint then returns 503.

Mobile-upload handling carried over from the original:
* EXIF orientation is applied before inference (phone photos are often
  stored sideways with an orientation tag).
* HEIC works if `pillow-heif` is installed, else a clear 415.
* Photos are downscaled to HAZARD_MAX_DIM on the long edge first.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.schemas.hazard_detection import (
    BoundingBox,
    DetectionWithBox,
    HazardVerificationResponse,
)

# Stop Ultralytics from pip-installing packages at request time based on what
# a caller uploads. Must be set before ultralytics is imported.
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "True")

logger = logging.getLogger(__name__)

EXPECTED_CLASSES = {"road_crack", "active_mudslide", "blocked_road", "normal"}
BOX_COLOR_PRIMARY = (0, 0, 255)  # red (BGR) -- top detection
BOX_COLOR_SECONDARY = (0, 165, 255)  # orange -- other detections
NORMAL_COLOR = (0, 200, 0)  # green -- "verified clear"


class ModelNotLoadedError(RuntimeError):
    pass


class ImageDecodeError(ValueError):
    pass


@dataclass
class _Detection:
    hazard_class: str
    confidence: float
    bbox: tuple[int, int, int, int]
    bbox_norm: tuple[float, float, float, float]

    def box(self) -> BoundingBox:
        x1, y1, x2, y2 = self.bbox
        nx1, ny1, nx2, ny2 = (round(v, 4) for v in self.bbox_norm)
        return BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2, x1_norm=nx1, y1_norm=ny1, x2_norm=nx2, y2_norm=ny2
        )


class HazardDetectorService:
    """One instance lives on `app.state.hazard_detector`."""

    def __init__(self, max_dim: int = 1280) -> None:
        self.model: Any = None
        self.class_names: dict[int, str] = {}
        self.max_dim = max_dim
        self.heic_support = False
        self.load_error: str | None = None
        # Ultralytics models aren't documented as thread-safe; requests run in
        # a threadpool, so serialise inference.
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Hazard model weights not found at '{path}'. Train with "
                f"ml/training/train_hazard_yolov8.py or set HAZARD_MODEL_PATH."
            )
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
            self.heic_support = True
        except ImportError:
            pass

        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.class_names = dict(self.model.names)

        names = set(self.class_names.values())
        if names != EXPECTED_CLASSES:
            logger.warning(
                "Hazard model classes %s differ from expected %s; responses will use "
                "the checkpoint's class names.",
                sorted(names),
                sorted(EXPECTED_CLASSES),
            )

    def try_load(self, path: Path) -> None:
        try:
            self.load(path)
            logger.info(
                "Hazard model loaded from %s (classes: %s)", path, list(self.class_names.values())
            )
        except Exception as exc:
            self.load_error = str(exc)
            logger.warning("Hazard detector unavailable: %s", exc)

    # ------------------------------------------------------------------ #
    # Public entry point (CPU/GPU-bound and synchronous -- call from a threadpool)
    # ------------------------------------------------------------------ #
    def verify(
        self,
        raw_bytes: bytes,
        content_type: str | None,
        filename: str | None,
        confidence: float,
        iou: float,
    ) -> HazardVerificationResponse:
        if not self.is_loaded:
            raise ModelNotLoadedError(self.load_error or "Hazard model is not loaded.")

        img_bgr = self._decode(raw_bytes, content_type)
        h, w = img_bgr.shape[:2]

        start = time.perf_counter()
        with self._lock:
            detections = self._infer(img_bgr, conf=confidence, iou=iou)
        inference_ms = (time.perf_counter() - start) * 1000

        annotated_uri = self._encode_jpeg_data_uri(self._annotate(img_bgr, detections))

        if detections:
            top = detections[0]
            hazard_class, top_conf, bbox = top.hazard_class, top.confidence, top.box()
        else:
            # Nothing cleared the threshold -> treat as verified clear.
            hazard_class, top_conf, bbox = "normal", 1.0, None

        return HazardVerificationResponse(
            filename=filename,
            image_width=w,
            image_height=h,
            hazard_class=hazard_class,
            confidence=round(top_conf, 4),
            hazard_detected=hazard_class != "normal",
            bounding_box=bbox,
            all_detections=[
                DetectionWithBox(
                    hazard_class=d.hazard_class,
                    confidence=round(d.confidence, 4),
                    bounding_box=d.box(),
                )
                for d in detections
            ],
            annotated_image=annotated_uri,
            model_classes=list(self.class_names.values()),
            confidence_threshold=confidence,
            inference_ms=round(inference_ms, 1),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _decode(self, raw_bytes: bytes, content_type: str | None) -> Any:
        import cv2
        import numpy as np
        from PIL import Image, ImageOps

        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as exc:
            hint = ""
            if content_type and "heic" in content_type.lower() and not self.heic_support:
                hint = (
                    " HEIC upload detected but pillow-heif isn't installed server-side; "
                    "install it or convert to JPEG/PNG before uploading."
                )
            raise ImageDecodeError(f"Could not decode the upload as an image.{hint}") from exc

        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        longest = max(h, w)
        if longest > self.max_dim:
            scale = self.max_dim / longest
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return bgr

    def _infer(self, img_bgr: Any, conf: float, iou: float) -> list[_Detection]:
        result = self.model.predict(source=img_bgr, conf=conf, iou=iou, verbose=False)[0]
        h, w = img_bgr.shape[:2]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                _Detection(
                    hazard_class=self.class_names.get(cls_id, str(cls_id)),
                    confidence=float(box.conf.item()),
                    bbox=(x1, y1, x2, y2),
                    bbox_norm=(x1 / w, y1 / h, x2 / w, y2 / h),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _annotate(img_bgr: Any, detections: list[_Detection]) -> Any:
        import cv2

        out = img_bgr.copy()
        if not detections:
            cv2.putText(
                out,
                "No hazard detected",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                NORMAL_COLOR,
                2,
                cv2.LINE_AA,
            )
            return out

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            color = BOX_COLOR_PRIMARY if i == 0 else BOX_COLOR_SECONDARY
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if i == 0 else 2)

            label = f"{det.hazard_class} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 8, th + 4)
            cv2.rectangle(out, (x1, label_y - th - 6), (x1 + tw + 6, label_y + 2), color, -1)
            cv2.putText(
                out,
                label,
                (x1 + 3, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return out

    @staticmethod
    def _encode_jpeg_data_uri(img_bgr: Any, quality: int = 85) -> str:
        import cv2

        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("Failed to encode annotated image as JPEG.")
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
