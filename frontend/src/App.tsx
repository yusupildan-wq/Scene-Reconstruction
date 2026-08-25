import { DragEvent, useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL, createJob, createProject, getJob, Job, listProjects, retryJob } from "./api/client";
import GaussianSplatViewer from "./GaussianSplatViewer";
import SceneViewer from "./SceneViewer";
import "./styles.css";
import "./viewer-controls.css";

const STAGES = [
  ["uploading", "Uploading"], ["preparing_frames", "Preparing Frames"],
  ["vggt_geometry", "VGGT Geometry"], ["gaussian_optimization", "Gaussian Optimization"],
  ["finalizing", "Finalizing"],
] as const;
const LEGACY_STAGE: Record<string, string> = {
  pending: "preparing_frames", extracting_frames: "preparing_frames", dispatched: "vggt_geometry",
  running_sfm: "vggt_geometry", training: "gaussian_optimization", complete: "finalizing",
};
const ACTIVE_JOB_KEY = "scene-reconstruction-active-job";

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewerUrl, setViewerUrl] = useState<string | null>(null);
  const [viewerLoaded, setViewerLoaded] = useState(false);
  const [experimentScene, setExperimentScene] = useState("v3_scene_high.ply");
  const handleViewerLoaded = useCallback(() => setViewerLoaded(true), []);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => {
    const savedJobId = localStorage.getItem(ACTIVE_JOB_KEY);
    if (!savedJobId) return;
    getJob(savedJobId).then((saved) => {
      setJob(saved);
      if (saved.status === "complete" && saved.scene_url) setViewerUrl(`${API_BASE_URL}${saved.scene_url}`);
    }).catch(() => localStorage.removeItem(ACTIVE_JOB_KEY));
  }, []);
  useEffect(() => {
    if (!job || job.status === "complete" || job.status === "failed") return;
    const timer = window.setInterval(async () => {
      try {
        const next = await getJob(job.id); setJob(next);
        if (next.status === "complete") {
          if (!next.scene_url) { setError("Reconstruction finished, but the viewer artifact is missing."); return; }
          setViewerUrl(`${API_BASE_URL}${next.scene_url}`);
        }
      } catch { /* transient polling failures should not discard the job */ }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  function choose(next: File | null) {
    setError(null);
    if (!next) return;
    if (!next.type.startsWith("video/")) { setError("Choose an MP4 or another video file."); return; }
    if (preview) URL.revokeObjectURL(preview);
    setFile(next); setPreview(URL.createObjectURL(next)); setJob(null); setUploadPercent(0);
  }
  function drop(event: DragEvent) { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files[0] ?? null); }
  async function submit() {
    if (!file) return;
    setUploading(true); setError(null); setUploadPercent(0);
    try {
      const projects = await listProjects();
      const project = projects[0] ?? await createProject("My room reconstructions");
      const created = await createJob(project.id, file, setUploadPercent);
      setJob(created); localStorage.setItem(ACTIVE_JOB_KEY, created.id); setUploadPercent(100);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setUploading(false); }
  }
  async function retry() {
    if (!job) return;
    setError(null);
    try { const retried = await retryJob(job.id); setJob(retried); localStorage.setItem(ACTIVE_JOB_KEY, retried.id); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }
  function reset() { localStorage.removeItem(ACTIVE_JOB_KEY); setViewerLoaded(false); setViewerUrl(null); setJob(null); setFile(null); setPreview(null); setError(null); }

  if (new URLSearchParams(window.location.search).has("experiments")) return <div className="viewer-shell">
    {experimentScene.endsWith(".ply") ? <GaussianSplatViewer key={experimentScene} sceneUrl={`/${experimentScene}`} />
      : <SceneViewer key={experimentScene} sceneUrl={`/${experimentScene}`} />}
    <div className="viewer-bar"><select value={experimentScene} onChange={(e) => setExperimentScene(e.target.value)}>
      <option value="v3_scene_high.ply">V3 High · VGGT + gsplat (3.43M)</option>
      <option value="v3_scene.ply">V3 Baseline · VGGT + gsplat (1.68M)</option><option value="dust3r_scene.ply">V1 · DUSt3R + gsplat</option>
      <option value="dust3r_scene.bin">V1 point viewer</option><option value="real_scene.json">V2 · COLMAP</option>
      <option value="demo_scene.json">Synthetic demo</option></select><button onClick={() => { window.location.search = ""; }}>Upload a room</button></div>
  </div>;

  if (viewerUrl) return <div className="viewer-shell">
    <GaussianSplatViewer sceneUrl={viewerUrl} onLoaded={handleViewerLoaded} />
    <div className="viewer-bar"><span>{viewerLoaded ? "Reconstruction complete" : "Opening your room…"}</span><button onClick={reset}>New reconstruction</button></div>
  </div>;

  const current = uploading ? "uploading" : LEGACY_STAGE[job?.status ?? ""] ?? job?.status;
  const currentIndex = STAGES.findIndex(([key]) => key === current);
  const progress = uploading ? Math.max(2, Math.round(uploadPercent * .08)) : job?.progress_percent ?? 0;
  const processing = uploading || (!!job && !["complete", "failed"].includes(job.status));

  return <main className="page">
    <header><div className="mark">S</div><span>Scene Reconstruction</span><a href="/?experiments=1">Experiments</a></header>
    <section className="card">
      {!processing && !job && <>
        <div className="intro"><span className="eyebrow">ROOM TO 3D</span><h1>Walk through your room again.</h1>
          <p>Upload a steady room video. We’ll reconstruct it into an interactive 3D scene.</p></div>
        <div className={`dropzone ${dragging ? "dragging" : ""}`} onDragEnter={() => setDragging(true)}
          onDragLeave={() => setDragging(false)} onDragOver={(e) => e.preventDefault()} onDrop={drop}
          onClick={() => inputRef.current?.click()}>
          <input ref={inputRef} hidden type="file" accept="video/mp4,video/quicktime,video/webm" onChange={(e) => choose(e.target.files?.[0] ?? null)} />
          <div className="upload-icon">↑</div><strong>Drop your room video here</strong><span>or click to choose a file</span>
        </div>
        {preview && file && <div className="preview"><video src={preview} controls preload="metadata" />
          <div><strong>{file.name}</strong><span>{(file.size / 1048576).toFixed(1)} MB</span></div></div>}
        {file && <button className="primary" onClick={submit}>Reconstruct room</button>}
      </>}

      {(processing || job) && <div className="progress-view">
        <div className="spinner-or-check">{job?.status === "failed" ? "!" : "◌"}</div>
        <span className="eyebrow">{job?.status === "failed" ? "RECONSTRUCTION PAUSED" : "BUILDING YOUR SCENE"}</span>
        <h2>{job?.status === "failed" ? "Something interrupted the reconstruction" : job?.stage_detail || "Uploading your video"}</h2>
        <p>{job?.status === "failed" ? "Your completed work is saved. Retry continues from the last reusable stage." : "You can leave this tab open while we turn the video into a navigable room."}</p>
        {job?.status !== "failed" && <><div className="progress-track"><i style={{ width: `${progress}%` }} /></div><b className="percent">{progress}%</b>
          <ol className="stages">{STAGES.map(([key, label], i) => <li key={key} className={i < currentIndex ? "done" : i === currentIndex ? "active" : ""}>
            <span>{i < currentIndex ? "✓" : i + 1}</span><em>{label}</em></li>)}</ol></>}
        {job?.status === "failed" && <button className="primary" onClick={retry}>Retry from saved progress</button>}
      </div>}
      {(error || job?.error_message) && <div className="error">{error || job?.error_message}</div>}
    </section>
    <footer>Best results: move slowly, keep objects in view, and avoid abrupt turns.</footer>
  </main>;
}
