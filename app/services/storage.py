"""Photo storage for field reports.

Local-disk implementation for dev / single-node deploys: files go to
MEDIA_DIR and are served at MEDIA_URL_PREFIX. For multi-node production,
replace `save_photo` with an object-store upload (S3/GCS) that returns the
public URL -- callers only depend on the returned string.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.core.config import settings

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def save_photo(data: bytes, original_filename: str | None) -> str:
    """Write the photo under a random name and return its public URL path."""
    ext = Path(original_filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        ext = ".jpg"
    # Random name: never trust a client-supplied filename on disk.
    name = f"{uuid.uuid4().hex}{ext}"
    target_dir = settings.MEDIA_DIR / "reports"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / name).write_bytes(data)
    return f"{settings.MEDIA_URL_PREFIX.rstrip('/')}/reports/{name}"
