from fastapi import APIRouter

from app.api.v1.endpoints import hazards, health, risk, risk_zones, sensors

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(risk.router)
api_router.include_router(hazards.verify_router)
api_router.include_router(hazards.router)
api_router.include_router(sensors.router)
api_router.include_router(risk_zones.router)
