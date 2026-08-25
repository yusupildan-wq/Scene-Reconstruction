export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
export type JobStatus = "pending" | "extracting_frames" | "dispatched" | "running_sfm" |
  "training" | "preparing_frames" | "vggt_geometry" | "gaussian_optimization" |
  "finalizing" | "complete" | "failed";
export interface Project { id: string; name: string; created_at: string; }
export interface Job {
  id: string; project_id: string; status: JobStatus; stage_detail: string | null;
  error_message: string | null; frame_count: number | null; selected_frame_count: number | null;
  output_storage_key: string | null; camera_storage_key: string | null; progress_percent: number;
  scene_url: string | null; cameras_url: string | null; metrics: Record<string, unknown> | null;
  created_at: string; updated_at: string;
}
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const body = await response.text();
    try { throw new Error(JSON.parse(body).detail || body); }
    catch (error) { if (error instanceof SyntaxError) throw new Error(body || `${response.status} ${response.statusText}`); throw error; }
  }
  return response.json() as Promise<T>;
}
export const listProjects = () => request<Project[]>("/projects");
export const createProject = (name: string) => request<Project>("/projects", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
});
export const getJob = (id: string) => request<Job>(`/jobs/${id}`);
export const retryJob = (id: string) => request<Job>(`/jobs/${id}/retry`, { method: "POST" });
export function createJob(projectId: string, video: File, onProgress: (percent: number) => void): Promise<Job> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}/projects/${projectId}/jobs`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(Math.round(e.loaded / e.total * 100));
    xhr.onerror = () => reject(new Error("Upload interrupted. Check your connection and try again."));
    xhr.onload = () => xhr.status >= 200 && xhr.status < 300
      ? resolve(JSON.parse(xhr.responseText) as Job)
      : reject(new Error((() => { try { return JSON.parse(xhr.responseText).detail; } catch { return xhr.responseText; } })() || `Upload failed (${xhr.status})`));
    const form = new FormData(); form.append("video", video); xhr.send(form);
  });
}
