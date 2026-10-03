"""Align hazard_reports with what the field PWA actually sends.

FieldReportPWA.jsx submits `notes` and GPS `accuracy`, which the original
schema had nowhere to store, and offers "Rockfall", which the original
hazard_type CHECK rejected. Rockfall is human-reported only -- the YOLOv8
detector doesn't classify it.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("hazard_reports", sa.Column("notes", sa.Text(), nullable=True))
    op.add_column(
        "hazard_reports",
        sa.Column("location_accuracy_m", sa.Numeric(8, 2), nullable=True),
    )
    op.drop_constraint(op.f("chk_hazard_reports_hazard_type"), "hazard_reports", type_="check")
    op.create_check_constraint(
        op.f("chk_hazard_reports_hazard_type"),
        "hazard_reports",
        "hazard_type IN ('road_crack', 'active_mudslide', 'blocked_road', 'rockfall')",
    )


def downgrade() -> None:
    # Fails if any 'rockfall' rows exist -- delete or remap them first.
    op.drop_constraint(op.f("chk_hazard_reports_hazard_type"), "hazard_reports", type_="check")
    op.create_check_constraint(
        op.f("chk_hazard_reports_hazard_type"),
        "hazard_reports",
        "hazard_type IN ('road_crack', 'active_mudslide', 'blocked_road')",
    )
    op.drop_column("hazard_reports", "location_accuracy_m")
    op.drop_column("hazard_reports", "notes")
