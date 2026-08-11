const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type JobStatus =
  | "pending"
  | "extracting_frames"
  | "dispatched"
  | "running_sfm"
  | "training"
  | "complete"
  | "failed";

export interface Project {
  id: string;
  name: string;
  created_at: string;
}

export interface Job {
  id: string;
  project_id: string;
  status: JobStatus;
  stage_detail: string | null;
  error_message: string | null;
  frame_count: number | null;
  selected_frame_count: number | null;
  output_storage_key: string | null;
  metrics: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export function listProjects(): Promise<Project[]> {
  return request("/projects");
}

export function createProject(name: string): Promise<Project> {
  return request("/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export function listJobs(projectId: string): Promise<Job[]> {
  return request(`/projects/${projectId}/jobs`);
}

export function createJob(projectId: string, video: File): Promise<Job> {
  const formData = new FormData();
  formData.append("video", video);
  return request(`/projects/${projectId}/jobs`, {
    method: "POST",
    body: formData,
  });
}

export function getJob(jobId: string): Promise<Job> {
  return request(`/jobs/${jobId}`);
}
