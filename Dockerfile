# syntax=docker/dockerfile:1
# GeoAlert NER API. Build arg EXTRAS picks which ML stacks are installed:
#   docker build --build-arg EXTRAS=risk .          # no torch, much smaller image
#   docker build --build-arg EXTRAS=risk,vision .   # default: everything
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

# libgl/libglib: OpenCV runtime deps (pulled in by ultralytics even with -headless).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS deps
ARG EXTRAS=risk,vision
# Install dependencies in their own layer so code changes don't reinstall torch.
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py \
    && pip install ".[${EXTRAS}]" \
    && pip uninstall -y geoalert-ner

FROM deps AS runtime
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser alembic.ini ./
# Model weights are mounted at runtime (see docker-compose.yml) or baked in by
# uncommenting the next line. Keeping them out of the image keeps builds fast.
# COPY --chown=appuser:appuser ml/models ./ml/models
RUN mkdir -p media && chown appuser:appuser media
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)" || exit 1

# One worker per container; scale with replicas. Each worker loads its own
# copy of the ML models, so more workers = more RAM.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
