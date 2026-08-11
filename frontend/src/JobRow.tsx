import { useEffect, useState } from "react";
import { getJob, Job } from "./api/client";

const TERMINAL_STATUSES = new Set(["complete", "failed"]);
const POLL_INTERVAL_MS = 2000;

export default function JobRow({ initialJob }: { initialJob: Job }) {
  const [job, setJob] = useState(initialJob);

  useEffect(() => {
    if (TERMINAL_STATUSES.has(job.status)) return;
    const timer = setInterval(async () => {
      try {
        setJob(await getJob(job.id));
      } catch (err) {
        console.error("Failed to poll job status", err);
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [job.id, job.status]);

  return (
    <li style={{ marginBottom: 12, padding: 12, border: "1px solid #ddd", borderRadius: 6 }}>
      <div>
        <strong>{job.id.slice(0, 8)}</strong> — <span>{job.status}</span>
        {job.stage_detail && <span> ({job.stage_detail})</span>}
      </div>
      {job.frame_count != null && (
        <div>
          frames: {job.frame_count} seen, {job.selected_frame_count} selected after blur/redundancy filtering
        </div>
      )}
      {job.error_message && <div style={{ color: "crimson" }}>error: {job.error_message}</div>}
      {job.status === "complete" && !job.output_storage_key && (
        <div>complete, but no viewer artifact yet (V0 viewer not wired up)</div>
      )}
    </li>
  );
}
