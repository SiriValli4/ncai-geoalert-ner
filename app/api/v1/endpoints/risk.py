from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import RiskModel
from app.schemas.risk import RiskRequest, RiskResponse
from app.services.risk_model import ModelNotLoadedError

router = APIRouter(prefix="/risk", tags=["risk"])


@router.post("/predict", response_model=RiskResponse, summary="Predict landslide susceptibility")
def predict_risk(payload: RiskRequest, model: RiskModel) -> RiskResponse:
    # Sync `def`: FastAPI runs it in a threadpool, keeping CPU-bound
    # inference off the event loop.
    try:
        return model.predict(payload)
    except ModelNotLoadedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Feature mismatch with loaded model: {exc}"
        ) from exc
