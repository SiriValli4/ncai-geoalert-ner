# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

GeoAlert NER is a landslide early-warning system for the North Eastern Region of India.

| Area | Path | Stack |
|---|---|---|
| Core API | `app/` | FastAPI, SQLAlchemy 2 (async), GeoAlchemy2, PostGIS |
| DB migrations | `alembic/` | Alembic (async env) |
| ML training | `ml/training/` | scikit-learn / XGBoost, Ultralytics YOLOv8 |
| Model artifacts | `ml/models/` | `.joblib` risk model, `.pt` YOLO weights |
| Training data | `data/` | CSV |
| SMS alerts | `services/alerts/` | Node/Express + Twilio (mock mode without credentials) |
| Web UI | `frontend/` | React 18 (CRA), Leaflet |
| Superseded code | `legacy/` | Reference only. Never edit or import from it. |

## Commands

```bash
# Setup (Python 3.11-3.12; ultralytics/torch lag behind the newest Python)
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[risk,dev]"          # add ,vision for YOLO (pulls torch), ,heic for iPhone photos
cp .env.example .env

# Database
docker compose up -d db
alembic upgrade head
alembic revision --autogenerate -m "describe change"   # then READ and edit the file

# Run / test / lint
uvicorn app.main:app --reload         # docs at http://localhost:8000/docs
pytest
ruff check . && ruff format .
mypy app

# Full stack
docker compose up --build

# Frontend (dashboard fetches live risk from REACT_APP_API_BASE_URL, see frontend/.env.example)
cd frontend && npm start
CI=true npx react-scripts test --watchAll=false
```

## Architecture rules

Requests flow in one direction: **endpoint → repository / service → model**.

- `app/api/v1/endpoints/`: HTTP only. Parse and validate input, call a repository or service, map errors to status codes, and commit the transaction. No SQL, no ML code.
- `app/repositories/`: all database queries. Plain async functions taking an `AsyncSession` as the first argument. They `flush()` and never `commit()`; the endpoint owns the transaction.
- `app/services/`: domain logic with no HTTP dependencies (ML inference, storage, terrain lookup). Services must not import FastAPI.
- `app/schemas/`: Pydantic request/response models. ORM objects never leave an endpoint unconverted; always use `Schema.model_validate(orm_obj)`.
- `app/models/`: SQLAlchemy ORM. Every model must be imported in `app/models/__init__.py` or Alembic won't see it.
- `app/core/config.py`: the only place that reads environment variables. Add new settings there and to `.env.example`.

Dependencies are typed aliases in `app/api/deps.py` (`DbSession`, `RiskModel`, `HazardDetector`). Use them instead of writing `Depends(...)` inline.

## PostGIS conventions (read before writing spatial code)

- All geometry is **SRID 4326** (WGS84). Columns are `geometry(...)`, not `geography`.
- **Axis order is (x=longitude, y=latitude).** APIs expose `latitude`/`longitude` fields. Always build points with `make_point(lat, lng)` / `point_to_wkb(lat, lng)` from `app/repositories/spatial.py` / `app/schemas/geo.py`, never by hand.
- **Distances in metres:** wrap the column in `as_geography(Model.location)`. That exact form compiles to `CAST(location AS geography)`, which matches the `idx_*_location_geog` expression indexes. Any other cast silently causes a sequential scan. `tests/test_spatial.py` guards this.
- `ST_Distance` / `ST_DWithin` on raw `geometry(4326)` return **degrees**. Never use them for "within N km".
- Containment (`ST_Contains`, `ST_Intersects`) works on the raw geometry columns and uses the plain GIST indexes.
- `risk_zones.boundary` is `MultiPolygon`. Wrap inserts with `ST_Multi(...)` (see `repositories/risk_zones.py`).
- `risk_zones.updated_at` is maintained by a DB trigger. Don't set it in Python.
- `legacy/001_landslide_ews_schema.sql` documents the EXPLAIN ANALYZE reasoning behind the indexes.

## Migrations

- The schema is owned by Alembic. Never change tables by hand or with `Base.metadata.create_all()`.
- Autogenerate is a draft: always review it. It handles expression indexes, triggers and CHECK constraints poorly. Write those with `op.execute(...)` as in `0001`.
- asyncpg rejects multiple SQL statements in one `op.execute`. Use one call per statement.
- Constraint names follow `app/db/base.py::NAMING_CONVENTION`. A CHECK named `"foo"` on table `t` becomes `chk_t_foo`.
- In `op.*` calls, wrap an already-full constraint name in `op.f(...)`. Otherwise the convention is applied again (`chk_t_chk_t_foo`) and the migration fails on a real database.
- Check every migration offline before applying it: `alembic upgrade head --sql` and `alembic downgrade <rev>:<rev> --sql`.
- Never edit a migration that has been applied to a shared database. Add a new one.

## ML models

- Models load once at startup (`app/main.py` lifespan) into `app.state`. A missing file or missing extra is logged and the related endpoint returns **503**. The API still boots. Keep it that way.
- Inference is blocking. Use a sync `def` endpoint or `await run_in_threadpool(...)`. Never call `.predict()` directly inside `async def`.
- YOLO inference is serialised with a lock in `HazardDetectorService`. Don't remove it.
- ML imports (`numpy`, `pandas`, `cv2`, `ultralytics`, `shap`) are **lazy, inside methods**, so the core API installs without them.
- The risk model's terrain inputs come from `app/services/terrain.py`, currently a **deterministic stub**. Responses list stubbed fields in `estimated_inputs`. Don't present those predictions as authoritative until a real DEM/land-cover/geology lookup replaces the stub.
- To retrain, run `ml/training/train_*.py` and write output to `ml/models/`. Feature names in the artifact must match `RiskRequest` + `terrain.py`.
- Pickled sklearn models are tied to the sklearn version that wrote them (the current artifact was written with 1.9.0). Retrain after any sklearn minor-version upgrade, and treat `InconsistentVersionWarning` as a signal.

## Domain constants (keep consistent everywhere)

- Risk score: 0-100. Severity: `Low`/`Medium`/`High` in the DB; `LOW`/`MEDIUM`/`HIGH` in risk-model responses.
- Score bands: ≥ 70 is HIGH, ≥ 40 is MEDIUM. Defined in both `services/alerts/dispatchAlertsRoute.js` (SMS dispatch) and `frontend/src/riskBands.js` (map colours, priority queue); change both together. The dashboard bands live predictions by **score**, not the model's argmax `severity`, so the map agrees with dispatch. The two can differ near a band edge.
- Hazard types: `road_crack`, `active_mudslide`, `blocked_road` (YOLO classes, plus `normal` = clear) and `rockfall` (human-reported only). The PWA's display labels are mapped in `app/schemas/hazard_report.py::PWA_LABEL_TO_HAZARD_TYPE`.
- Report status: `pending` → `verified` / `rejected` → `resolved`.

## API conventions

- Versioned under `API_PREFIX` (default `/api/v1`). `POST /api/submit-report` is a deprecated alias kept for the field PWA. Remove it once the frontend posts to `/api/v1/hazard-reports`.
- The field PWA sends **multipart form data with camelCase keys** (`hazardType`, `capturedAt`) and empty strings for missing values. Endpoints serving it must tolerate both.
- Errors: `HTTPException` with a human-readable `detail`. Unhandled exceptions are logged and returned as a generic 500. Never leak tracebacks or SQL.
- Uploads: enforce `settings.max_upload_bytes` and never use a client-supplied filename on disk (`services/storage.py`).

## Style

- Python ≥ 3.11, `from __future__ import annotations`, full type hints, `ruff` (line length 100).
- Use `logging.getLogger(__name__)`. No `print()`.
- Comments explain *why*, not *what*. Much of this codebase carries hard-won reasoning in comments (EXIF rotation, SHAP projection, index choices). Preserve it when refactoring.

## Testing

- `pytest` runs without a database. API tests override `get_db` with a stub that fails loudly if touched (`tests/conftest.py`).
- Tests for real spatial queries need PostGIS: `docker compose up -d db && alembic upgrade head`. Mark them so they can be skipped when no DB is available.
- ML tests skip automatically when the artifact or extra is missing.

## Don'ts

- Don't commit `.env`, `media/`, `runs/` or new large model weights without discussing storage (Git LFS / object store).
- Don't add code to `legacy/`.
- Don't call Twilio for real in tests. `services/alerts` runs in mock mode when `TWILIO_*` is unset.
