"""per-job compute execution mode

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("execution_mode", sa.String(32), nullable=False, server_default="runpod"))


def downgrade() -> None:
    op.drop_column("jobs", "execution_mode")
