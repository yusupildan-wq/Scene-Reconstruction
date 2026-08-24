"""product reconstruction stages and artifacts

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for value in ("preparing_frames", "vggt_geometry", "gaussian_optimization", "finalizing"):
        op.execute(f"ALTER TYPE jobstatus ADD VALUE IF NOT EXISTS '{value}'")
    op.add_column("jobs", sa.Column("camera_storage_key", sa.String(1024), nullable=True))
    op.add_column("jobs", sa.Column("progress_percent", sa.Integer(), nullable=False, server_default="0"))
    op.add_column(
        "jobs",
        sa.Column("stage_artifacts", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )


def downgrade() -> None:
    op.drop_column("jobs", "stage_artifacts")
    op.drop_column("jobs", "progress_percent")
    op.drop_column("jobs", "camera_storage_key")
