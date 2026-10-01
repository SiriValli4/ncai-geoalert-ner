# legacy/

The original standalone services, kept for reference. **Superseded by `app/`. Do not extend these files.**

| File | Replaced by |
|---|---|
| `landslide_risk_api.py` (port 8000, `POST /predict-risk`) | `POST /api/v1/risk/predict` (`app/services/risk_model.py`) |
| `hazard_detection_api.py` (port 8001, `POST /verify-hazard-image`) | `POST /api/v1/hazards/verify-image` (`app/services/hazard_detector.py`) |
| `001_landslide_ews_schema.sql` | `alembic/versions/0001_initial_postgis_schema.py` (+ `0002`) |
| `requirements.txt` | `pyproject.toml` extras: `risk`, `vision`, `heic` |

The request and response bodies of both ML endpoints are unchanged; only the paths moved. Keep the SQL file: it is still the best explanation of why the geography expression indexes exist.

Once nothing depends on the old ports/paths, this folder can be deleted.
