from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import DbSession, HazardDetector
from app.core.config import settings
from app.models import HazardType, ReportStatus
from app.repositories import hazard_reports as repo
from app.schemas.hazard_detection import HazardVerificationResponse
from app.schemas.hazard_report import (
    HazardReportInZone,
    HazardReportRead,
    HazardReportStatusUpdate,
    normalize_hazard_type,
)
from app.services import hazard_detector as detector_mod
from app.services.storage import save_photo

router = APIRouter(prefix="/hazard-reports", tags=["hazard reports"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _read_image_upload(file: UploadFile) -> bytes:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Expected an image upload, got '{file.content_type}'.",
        )
    # Read one byte past the limit so oversize uploads are rejected without
    # buffering an arbitrarily large body.
    data = await file.read(settings.max_upload_bytes + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Upload exceeds the {settings.MAX_UPLOAD_MB} MB limit.",
        )
    return data


def _optional_float(raw: str | None, field: str) -> float | None:
    # The PWA sends "" for values it couldn't capture.
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise HTTPException(422, f"{field} must be a number") from exc


def _parse_captured_at(raw: str | None) -> datetime | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        dt = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, "capturedAt must be an ISO-8601 timestamp") from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Field report submission (contract: frontend/src/FieldReportPWA.jsx)
# --------------------------------------------------------------------------- #
@router.post(
    "",
    response_model=HazardReportRead,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a field hazard report (multipart, as sent by the field PWA)",
)
async def submit_report(
    db: DbSession,
    hazard_type: str = Form(..., alias="hazardType"),
    latitude: float = Form(..., ge=-90, le=90),
    longitude: float = Form(..., ge=-180, le=180),
    notes: str | None = Form(None, max_length=4000),
    accuracy: str | None = Form(None, description="GPS accuracy in metres."),
    captured_at: str | None = Form(None, alias="capturedAt"),
    photo: UploadFile | None = File(None),
) -> HazardReportRead:
    try:
        hz = normalize_hazard_type(hazard_type)
    except ValueError as exc:
        raise HTTPException(422, f"Unknown hazardType '{hazard_type}'") from exc

    photo_url = None
    if photo is not None and photo.filename:
        data = await _read_image_upload(photo)
        photo_url = await run_in_threadpool(save_photo, data, photo.filename)

    report = await repo.create_report(
        db,
        hazard_type=hz,
        latitude=latitude,
        longitude=longitude,
        reported_at=_parse_captured_at(captured_at),
        notes=(notes or "").strip() or None,
        accuracy_m=_optional_float(accuracy, "accuracy"),
        photo_url=photo_url,
    )
    await db.commit()
    return HazardReportRead.model_validate(report)


@router.get("", response_model=list[HazardReportRead], summary="List hazard reports")
async def list_reports(
    db: DbSession,
    status_: ReportStatus | None = Query(None, alias="status"),
    hazard_type: HazardType | None = None,
    lat: float | None = Query(None, ge=-90, le=90),
    lng: float | None = Query(None, ge=-180, le=180),
    radius_m: float | None = Query(None, gt=0, le=200_000),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[HazardReportRead]:
    near = None
    if any(v is not None for v in (lat, lng, radius_m)):
        if lat is None or lng is None or radius_m is None:
            raise HTTPException(
                422,
                "lat, lng and radius_m must be given together",
            )
        near = (lat, lng)
    rows = await repo.list_reports(
        db,
        status=status_,
        hazard_type=hazard_type,
        near=near,
        radius_m=radius_m,
        limit=limit,
        offset=offset,
    )
    return [HazardReportRead.model_validate(r) for r in rows]


@router.get(
    "/dispatch-candidates",
    response_model=list[HazardReportInZone],
    summary="Pending reports inside High-severity risk zones",
)
async def dispatch_candidates(
    db: DbSession, limit: int = Query(200, ge=1, le=2000)
) -> list[HazardReportInZone]:
    rows = await repo.pending_in_high_risk_zones(db, limit)
    return [
        HazardReportInZone(report=HazardReportRead.model_validate(r), zone_id=zid, zone_name=zname)
        for r, zid, zname in rows
    ]


@router.get("/{report_id}", response_model=HazardReportRead, summary="Get one report")
async def get_report(report_id: uuid.UUID, db: DbSession) -> HazardReportRead:
    report = await repo.get_report(db, report_id)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    return HazardReportRead.model_validate(report)


@router.patch(
    "/{report_id}/status", response_model=HazardReportRead, summary="Verify/resolve/reject"
)
async def update_report_status(
    report_id: uuid.UUID, payload: HazardReportStatusUpdate, db: DbSession
) -> HazardReportRead:
    report = await repo.get_report(db, report_id)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    await repo.update_status(db, report, payload.status)
    await db.commit()
    return HazardReportRead.model_validate(report)


# --------------------------------------------------------------------------- #
# AI photo verification (YOLOv8)
# --------------------------------------------------------------------------- #
verify_router = APIRouter(prefix="/hazards", tags=["hazard detection"])


@verify_router.post(
    "/verify-image",
    response_model=HazardVerificationResponse,
    summary="Detect road hazards in a field photo",
)
async def verify_hazard_image(
    detector: HazardDetector,
    file: UploadFile = File(..., description="Field photo (JPEG/PNG, HEIC if supported)."),
    confidence: float = Query(
        settings.HAZARD_CONF_DEFAULT, ge=0.0, le=1.0, description="Min confidence."
    ),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS."),
) -> HazardVerificationResponse:
    if not detector.is_loaded:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detector.load_error or "Hazard model is not loaded.",
        )
    data = await _read_image_upload(file)
    try:
        # Decode + inference are blocking; keep them off the event loop.
        return await run_in_threadpool(
            detector.verify, data, file.content_type, file.filename, confidence, iou
        )
    except detector_mod.ImageDecodeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
