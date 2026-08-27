import asyncio
import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import BackgroundTasks

from app.models import Job, JobStatus
from app.routers.jobs import retry_job


class RetryJobTests(unittest.TestCase):
    def test_retry_resets_stale_progress_percent(self):
        """A prior failed attempt can leave progress_percent stuck high (e.g. a
        cleanup-path report fired before the failure). Retry must reset it so the
        resumed run's progress bar reflects the stage it actually restarts from,
        not a stale high-water mark from the run that failed."""
        job = Job(
            id=uuid.uuid4(), project_id=uuid.uuid4(), status=JobStatus.FAILED,
            progress_percent=99, stage_detail="Terminating temporary RunPod GPU",
            error_message="TimeoutError: RunPod did not become reachable before the startup timeout",
            runpod_job_id="pod-1", input_storage_key="k", execution_mode="runpod",
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        session = SimpleNamespace(get=AsyncMock(return_value=job), commit=AsyncMock(), refresh=AsyncMock())
        background_tasks = BackgroundTasks()
        with patch("app.routers.jobs.run_pipeline"):
            asyncio.run(retry_job(job.id, background_tasks, session))
        self.assertEqual(job.progress_percent, 0)
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertIsNone(job.runpod_job_id)


if __name__ == "__main__":
    unittest.main()
