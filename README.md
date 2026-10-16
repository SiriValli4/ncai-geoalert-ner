# GeoAlert NER

**A landslide early-warning system for the North Eastern Region of India.**

GeoAlert NER combines machine-learning risk prediction, AI photo verification, crowdsourced field reports and multilingual SMS alerts into one platform for disaster-management teams working across Assam, Meghalaya, Sikkim, Arunachal Pradesh, Mizoram, Manipur and Nagaland.

| Operations dashboard | Offline-first field reporting |
|---|---|
| ![Dashboard with district risk zones](frontend/dashboard_initial.png) | ![Field report queued while offline](frontend/pwa_offline_queued_fixed.png) |

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Local setup](#local-setup)
- [API endpoints](#api-endpoints)
- [Testing](#testing)
- [Staging deployment](#staging-deployment)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)

---

## What it does

| Capability | How |
|---|---|
| **Predict landslide risk** | A RandomForest model scores any location 0–100 from 24h rainfall, soil moisture and slope, and explains its top contributing factors with SHAP. |
| **Verify hazard photos** | A YOLOv8 detector checks field photos for road cracks, active mudslides and blocked roads, and returns the image with the hazards boxed. |
| **Collect field reports** | A Progressive Web App lets ground officers file geotagged reports with photos. Reports queue on the device when there's no signal and sync automatically when it returns. |
| **Monitor the region** | A live map shows district risk zones coloured by the AI model, citizen reports, road status and a rainfall heatmap, plus an emergency priority queue. |
| **Store geospatial data** | PostGIS holds sensor telemetry, hazard reports and risk-zone polygons, indexed for "within N km" and "which zone contains this point" queries. |
| **Alert citizens** | When a district's score reaches HIGH (≥ 70), SMS alerts go out through Twilio in English, Assamese, Bengali and Manipuri. |

---

## Architecture

```mermaid
flowchart LR
    subgraph Clients
        DASH["React dashboard<br/>(Leaflet map)"]
        PWA["Field report PWA<br/>(offline queue)"]
    end

    subgraph API["FastAPI service (app/)"]
        EP["Endpoints<br/>/api/v1/*"]
        RISK["Risk model service<br/>RandomForest + SHAP"]
        YOLO["Hazard detector<br/>YOLOv8"]
        REPO["Repositories<br/>(spatial queries)"]
    end

    DB[("PostgreSQL + PostGIS")]
    MEDIA[("Photo storage<br/>media/")]
    ALERTS["Alerts service (Node)<br/>services/alerts/"]
    TWILIO["Twilio SMS"]

    DASH -- "POST /risk/predict" --> EP
    PWA -- "POST /hazard-reports" --> EP
    EP --> RISK
    EP --> YOLO
    EP --> REPO --> DB
    EP --> MEDIA
    ALERTS --> TWILIO
```

Inside the API, requests flow in one direction: **endpoint → repository/service → model**.

- Endpoints only handle HTTP.
- Repositories own all SQL.
- Services own the ML inference and file storage, and know nothing about HTTP.

The ML models load once at startup. If a model file or its libraries are missing, the API still starts and only that model's endpoint returns `503`. Design rules are documented in [`CLAUDE.md`](CLAUDE.md).

---

## Tech stack

| Layer | Technology |
|---|---|
| **API** | Python 3.11–3.12, FastAPI, Pydantic v2, Uvicorn |
| **Database** | PostgreSQL 16 + PostGIS 3.4, SQLAlchemy 2 (async, asyncpg), GeoAlchemy2, Alembic |
| **Risk model** | scikit-learn RandomForest (XGBoost optional), SHAP explanations |
| **Vision model** | Ultralytics YOLOv8, OpenCV, Pillow (optional HEIC support) |
| **Frontend** | React 18 (Create React App), React-Leaflet, Leaflet.heat, Tailwind CSS, IndexedDB (`idb`) |
| **Alerts** | Node.js, Express 5, Twilio |
| **Tooling** | Docker / Docker Compose, pytest, Jest, ruff, mypy |

---

## Project structure

```
geoalert-ner/
├── app/                      FastAPI application
│   ├── api/v1/endpoints/     HTTP routes (risk, hazards, sensors, risk-zones, health)
│   ├── core/                 Settings (env vars) and logging
│   ├── db/                   Async engine, session, base model
│   ├── models/               SQLAlchemy ORM: sensors_data, hazard_reports, risk_zones
│   ├── repositories/         All database / PostGIS queries
│   ├── schemas/              Pydantic request & response models
│   └── services/             Risk model, YOLO detector, photo storage, terrain lookup
├── alembic/versions/         Database migrations
├── frontend/                 React dashboard + field report PWA
├── ml/
│   ├── models/               Trained artifacts (.joblib, .pt)
│   └── training/             Training scripts for both models
├── data/                     Training dataset (CSV)
├── services/alerts/          Node/Express SMS dispatch service
├── tests/                    Backend tests (run without a database)
├── legacy/                   Original standalone APIs, kept for reference only
├── docker-compose.yml        PostGIS + API + alerts, for local development
├── Dockerfile                API container image
├── pyproject.toml            Python dependencies and tool config
└── CLAUDE.md                 Engineering guidelines and conventions
```

---

## Local setup

### Prerequisites

- **Python 3.11 or 3.12.** PyTorch/Ultralytics don't support the newest Python releases yet.
- **Node.js 18+** and npm
- **Docker Desktop** for the PostGIS database

### Option A: everything in Docker

```bash
git clone https://github.com/SiriValli4/geoalert-ner.git
cd geoalert-ner

docker compose up -d db                          # start PostGIS
docker compose run --rm api alembic upgrade head # create the tables
docker compose up --build                        # API on :8000, alerts on :3001
```

Then start the frontend as in [step 3](#3-frontend) below.

> The first build installs PyTorch and takes a while. For a much smaller image without photo verification, run `API_EXTRAS=risk docker compose up --build`.

### Option B: run each part directly

#### 1. Database

```bash
docker compose up -d db
```

This starts PostgreSQL + PostGIS on `localhost:5432` (user, password and database are all `geoalert`).

#### 2. Backend API

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e ".[risk,dev]"         # core API + risk model + dev tools
# optional: pip install -e ".[risk,vision,dev]"   adds YOLOv8 photo verification (large)
# optional: add ,heic to accept iPhone HEIC photos

cp .env.example .env                 # defaults work for local development
alembic upgrade head                 # create the database tables
uvicorn app.main:app --reload
```

- API: http://localhost:8000
- Interactive docs: **http://localhost:8000/docs**
- Readiness check (database + models): http://localhost:8000/api/v1/health/ready

#### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env.local           # optional: defaults to http://localhost:8000/api/v1
npm start
```

The dashboard opens at **http://localhost:3000** and fetches live risk predictions from the API. The header shows the connection state. If the API isn't running, the map falls back to built-in sample data and says so.

#### 4. SMS alerts service (optional)

```bash
cd services/alerts
npm install
cp .env.example .env                 # leave the Twilio values empty for mock mode
PORT=3001 npm start                  # Windows PowerShell: $env:PORT=3001; npm start
```

Use port 3001 so it doesn't clash with the React dev server on 3000. Without Twilio credentials, the service logs the messages it *would* send instead of sending them.

### Configuration

All backend settings are environment variables, documented in [`.env.example`](.env.example) and read only by `app/core/config.py`. The ones you're most likely to change:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://geoalert:geoalert@localhost:5432/geoalert` | Must use the `asyncpg` driver |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated origins allowed to call the API, e.g. your Vercel URL. `*` is accepted only when `ENVIRONMENT=local` |
| `ENVIRONMENT` | `local` | `local` / `staging` / `production`. `production` disables `/docs` |
| `RISK_MODEL_PATH` | `ml/models/landslide_risk_model.joblib` | Risk model artifact |
| `HAZARD_MODEL_PATH` | `ml/models/hazard_yolov8_demo.pt` | YOLOv8 weights |
| `MAX_UPLOAD_MB` | `15` | Photo upload size limit |
| `WEATHER_INGESTION_ENABLED` | `false` | Hourly OpenWeatherMap rainfall/humidity/temperature ingestion into `realtime_weather_logs`. Enable on **one** process only |
| `OPENWEATHER_API_KEY` | *(unset)* | OpenWeatherMap key. Without it ingestion logs a warning and stays off |
| `WEATHER_TARGETS` | `[]` | JSON list of `{name, latitude, longitude}` points to ingest |
| `REACT_APP_API_BASE_URL` | `http://localhost:8000/api/v1` | Frontend: API location (set at build time) |

---

## API endpoints

All endpoints are under `/api/v1`. Full request and response schemas are in the interactive docs at `/docs`.

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness: the process is up (never touches the database) |
| `GET` | `/health/ready` | Readiness: database + PostGIS reachable; reports which ML models loaded |

### Risk prediction

| Method | Path | Description |
|---|---|---|
| `POST` | `/risk/predict` | Landslide susceptibility (0–100), severity, class probabilities and the top factors behind the score |

```bash
curl -X POST http://localhost:8000/api/v1/risk/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 27.33, "longitude": 88.62, "rainfall_24h_mm": 185,
       "soil_moisture_percent": 82, "slope_degree": 38}'
```

```json
{
  "landslide_susceptibility_index": 85.2,
  "severity": "HIGH",
  "class_probabilities": { "LOW": 0.0379, "MEDIUM": 0.2201, "HIGH": 0.742 },
  "top_factors": [
    { "feature": "rainfall_24h_mm", "direction": "increases_risk",
      "explanation": "rainfall_24h_mm = 185.0 (heavy recent rainfall) pushes the susceptibility index higher (~14.2 pts)." }
  ],
  "estimated_inputs": ["land_cover_code", "lithology_stability_index", "elevation_m"],
  "explanation_method": "shap",
  "model_type": "RandomForestClassifier"
}
```

The response is abbreviated above. `estimated_inputs` lists the terrain values that were estimated rather than measured; see [Known limitations](#known-limitations).

### Hazard photo verification

| Method | Path | Description |
|---|---|---|
| `POST` | `/hazards/verify-image` | Upload a photo (multipart `file`). Returns the detected hazard class, confidence, bounding boxes and an annotated image. Optional query params: `confidence`, `iou` |

### Field hazard reports

| Method | Path | Description |
|---|---|---|
| `POST` | `/hazard-reports` | Submit a report (multipart: `hazardType`, `latitude`, `longitude`, `notes`, `accuracy`, `capturedAt`, `photo`) |
| `GET` | `/hazard-reports` | List reports. Filters: `status`, `hazard_type`, and `lat` + `lng` + `radius_m` for a radius search |
| `GET` | `/hazard-reports/dispatch-candidates` | Pending reports that fall inside High-severity risk zones |
| `GET` | `/hazard-reports/{report_id}` | Get one report |
| `PATCH` | `/hazard-reports/{report_id}/status` | Set status to `verified`, `rejected` or `resolved` |

`POST /api/submit-report` (no `/v1`) is a **deprecated alias** of `POST /hazard-reports`. It's kept so the current field PWA keeps working.

### Sensors

| Method | Path | Description |
|---|---|---|
| `POST` | `/sensors/readings` | Ingest a rainfall / soil-moisture reading |
| `GET` | `/sensors/latest` | Latest reading for every sensor |
| `GET` | `/sensors/nearby` | Readings within `radius_m` metres of `lat`/`lng`, nearest first |
| `GET` | `/sensors/{sensor_id}/readings` | Reading history for one sensor (`since`, `until`) |

### Risk zones

| Method | Path | Description |
|---|---|---|
| `GET` | `/risk-zones` | All zones as a GeoJSON FeatureCollection, ready for Leaflet (`severity` filter) |
| `POST` | `/risk-zones` | Create a zone from a GeoJSON Polygon or MultiPolygon |
| `GET` | `/risk-zones/containing` | Which zones contain the point `lat`/`lng` |

### Alerts service (separate Node service)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/dispatch-alerts` | Body: `district_name`, `risk_score`, optional `affected_highways`. Sends multilingual SMS only when `risk_score ≥ 70`; otherwise returns `dispatched: false` |

---

## Testing

```bash
# Backend: runs without a database or GPU
pytest
ruff check . && mypy app

# Frontend
cd frontend && CI=true npx react-scripts test --watchAll=false
```

The backend tests check the API's behaviour and validation, the geometry handling, and that distance queries compile to the exact SQL the database indexes expect. ML tests skip automatically when a model or its libraries aren't installed. The frontend tests cover the dashboard's live, offline and connection-lost states.

Database migrations can be checked without a database:

```bash
alembic upgrade head --sql
```

---

## Staging deployment

> **No staging environment exists yet.** This section describes the recommended setup, built from what the repository already provides: a `Dockerfile`, Alembic migrations, health probes, and environment-based config.

### Components

| Component | Recommended setup |
|---|---|
| **API** | The `Dockerfile` image on any container platform (e.g. AWS ECS/App Runner, Google Cloud Run, Azure Container Apps, Render, Fly.io). One worker per container; scale with replicas. |
| **Database** | Managed PostgreSQL 16 with the PostGIS extension enabled (AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL, or similar). |
| **Frontend** | `npm run build` with `REACT_APP_API_BASE_URL` pointing at the staging API, served as static files (S3 + CloudFront, Netlify, Vercel, or similar). |
| **Model files** | Bake into the image (uncomment the `COPY ml/models` line in the `Dockerfile`) or mount from a volume / object store. |
| **Photos** | Local disk works for a single instance only. Before running more than one replica, swap `app/services/storage.py` for object storage (S3/GCS). |
| **Alerts** | Deploy `services/alerts/` with **Twilio test credentials**, or none at all for mock mode, so staging never texts real people. |

### Release steps

1. **Build** the API image and push it to your container registry.
2. **Migrate:** run `alembic upgrade head` once against the staging database, as a one-off job **before** rolling out new containers.
3. **Deploy** the API with the staging configuration:

   ```bash
   ENVIRONMENT=staging
   DATABASE_URL=postgresql+asyncpg://<user>:<password>@<host>:5432/<db>
   CORS_ORIGINS=https://staging.your-domain.example
   RISK_MODEL_PATH=/srv/ml/models/landslide_risk_model.joblib
   HAZARD_MODEL_PATH=/srv/ml/models/hazard_yolov8_demo.pt
   ```

   Keep secrets in the platform's secret manager, never in the image or the repository.

4. **Configure probes:**
   - **liveness** → `GET /api/v1/health`
   - **readiness** → `GET /api/v1/health/ready`. It returns `503` until the database is reachable, so traffic waits for it.
5. **Build and deploy the frontend** with `REACT_APP_API_BASE_URL=https://<staging-api>/api/v1`.
6. **Smoke test:**
   - `GET /api/v1/health/ready` shows `"database": {"ok": true}` and the risk model loaded;
   - the dashboard header shows **Live AI risk**;
   - a test report submitted from the PWA appears in `GET /api/v1/hazard-reports`.

### Before exposing staging publicly

- **Add authentication.** The API currently has **no authentication**: anyone who can reach it can submit reports, change report status and create risk zones. Put it behind an auth layer (API gateway, OAuth/JWT, or at minimum IP allow-listing) before giving it a public URL.
- Add rate limiting on the upload endpoints.
- Set `CORS_ORIGINS` to the exact staging frontend origin, never `*`.

---

## Known limitations

These are deliberate placeholders, clearly marked in code. Read them before relying on any output:

- **Terrain inputs are estimated.** The risk model needs land cover, rock stability and elevation. Until a real GIS/DEM lookup replaces the stub in `app/services/terrain.py`, these are deterministic placeholders, and every prediction lists them in `estimated_inputs`.
- **Dashboard conditions are sample data.** The rainfall, soil moisture and slope sent for each district come from sample values in `LandslideDashboard.jsx`. The predictions are live; the weather isn't, yet. The next step is to feed them from `GET /sensors/nearby`.
- **Map boundaries are illustrative.** District polygons are simplified squares, not official administrative boundaries.
- **The YOLO model is a demo checkpoint.** `hazard_yolov8_demo.pt` should be retrained on real labelled field photos (see `ml/training/train_hazard_yolov8.py`).
- **There is no authentication** (see [above](#before-exposing-staging-publicly)).
- **Model files are tied to the scikit-learn version.** The risk model was saved with scikit-learn 1.9.0. Retrain it after upgrading scikit-learn.

---

## Contributing

1. Read [`CLAUDE.md`](CLAUDE.md). It covers the architecture rules, PostGIS conventions (axis order, metre-based distances), migration practices and domain constants.
2. Create a feature branch from `main`.
3. Make sure `pytest`, `ruff check .`, `mypy app` and the frontend tests pass.
4. Database changes go in a new Alembic migration. Never edit one that has already been applied.
5. Open a pull request.

Do not add code to `legacy/`. It's kept only as a reference for the original standalone services.
