"""initial schema: projects, jobs

Revision ID: 0001
Revises:
Create Date: 2026-08-10

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

job_status_enum = postgresql.ENUM(
    "pending",
    "extracting_frames",
    "dispatched",
    "running_sfm",
    "training",
    "complete",
    "failed",
    name="jobstatus",
)


def upgrade() -> None:
    # Not calling job_status_enum.create() explicitly here: op.create_table below
    # already creates the ENUM type as part of creating the "jobs" table's status
    # column, and Alembic's create_table does that unconditionally (no existence
    # check) -- calling .create() first caused a duplicate "CREATE TYPE" collision.
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("status", job_status_enum, nullable=False, server_default="pending"),
        sa.Column("stage_detail", sa.String(255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("input_storage_key", sa.String(1024), nullable=False),
        sa.Column("frame_count", sa.Integer(), nullable=True),
        sa.Column("selected_frame_count", sa.Integer(), nullable=True),
        sa.Column("output_storage_key", sa.String(1024), nullable=True),
        sa.Column("runpod_job_id", sa.String(255), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("projects")
    job_status_enum.drop(op.get_bind(), checkfirst=True)
