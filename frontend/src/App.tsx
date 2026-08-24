import { useEffect, useState } from "react";
import { createJob, createProject, listJobs, listProjects, Job, Project } from "./api/client";
import JobRow from "./JobRow";
import SceneViewer from "./SceneViewer";
import GaussianSplatViewer from "./GaussianSplatViewer";

export default function App() {
  const [activeTab, setActiveTab] = useState<"projects" | "demo">("demo");
  const [sceneChoice, setSceneChoice] = useState<
    | "demo_scene.json"
    | "real_scene.json"
    | "dust3r_scene.bin"
    | "dust3r_scene.ply"
    | "v3_scene.ply"
  >("v3_scene.ply");
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [newProjectName, setNewProjectName] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  useEffect(() => {
    listProjects().then(setProjects).catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedProjectId) {
      setJobs([]);
      return;
    }
    listJobs(selectedProjectId).then(setJobs).catch(console.error);
  }, [selectedProjectId]);

  async function handleCreateProject(event: React.FormEvent) {
    event.preventDefault();
    if (!newProjectName.trim()) return;
    const project = await createProject(newProjectName.trim());
    setProjects((prev) => [project, ...prev]);
    setSelectedProjectId(project.id);
    setNewProjectName("");
  }

  async function handleUpload(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedProjectId || !selectedFile) return;
    setUploadError(null);
    try {
      const job = await createJob(selectedProjectId, selectedFile);
      setJobs((prev) => [job, ...prev]);
      setSelectedFile(null);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    }
  }

  if (activeTab === "demo") {
    // Full-viewport, not the constrained page layout below -- the scene
    // should fill the screen the moment it loads, not sit in a small boxed
    // preview under a header. Controls float on top instead of pushing the
    // canvas down.
    return (
      <div style={{ position: "fixed", inset: 0, background: "#000" }}>
        {sceneChoice.endsWith(".ply") ? (
          <GaussianSplatViewer key={sceneChoice} sceneUrl={`/${sceneChoice}`} />
        ) : (
          <SceneViewer key={sceneChoice} sceneUrl={`/${sceneChoice}`} />
        )}
        <div
          style={{
            position: "absolute",
            top: 12,
            left: 12,
            background: "rgba(0,0,0,0.6)",
            color: "#fff",
            padding: 12,
            borderRadius: 8,
            fontFamily: "system-ui, sans-serif",
            maxWidth: 340,
          }}
        >
          <div style={{ marginBottom: 8, display: "flex", gap: 8 }}>
            <button onClick={() => setActiveTab("demo")} disabled>
              Demo Scene
            </button>
            <button onClick={() => setActiveTab("projects")}>Projects</button>
          </div>
          <select
            value={sceneChoice}
            onChange={(e) => setSceneChoice(e.target.value as typeof sceneChoice)}
          >
            <option value="v3_scene.ply">V3: VGGT + gsplat (1.68M Gaussians)</option>
            <option value="dust3r_scene.ply">V1: Full room (380k Gaussians, 64 cameras)</option>
            <option value="dust3r_scene.bin">My room (DUSt3R + 20k iterations, point-sprite viewer)</option>
            <option value="real_scene.json">My room (COLMAP + 6k iterations, V0)</option>
            <option value="demo_scene.json">Synthetic test scene (1000 iterations)</option>
          </select>
        </div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>Scene Reconstruction</h1>

      <div style={{ marginBottom: 24 }}>
        <button onClick={() => setActiveTab("demo")}>Demo Scene</button>{" "}
        <button onClick={() => setActiveTab("projects")} disabled>
          Projects
        </button>
      </div>

      <section style={{ marginBottom: 32 }}>
        <h2>Projects</h2>
        <form onSubmit={handleCreateProject} style={{ marginBottom: 12 }}>
          <input
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            placeholder="New project name"
          />
          <button type="submit">Create</button>
        </form>
        <select
          value={selectedProjectId ?? ""}
          onChange={(e) => setSelectedProjectId(e.target.value || null)}
        >
          <option value="">Select a project…</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </section>

      {selectedProjectId && (
        <section>
          <h2>Upload video</h2>
          <form onSubmit={handleUpload} style={{ marginBottom: 24 }}>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
            />
            <button type="submit" disabled={!selectedFile}>
              Upload &amp; start job
            </button>
          </form>
          {uploadError && <p style={{ color: "crimson" }}>{uploadError}</p>}

          <h2>Jobs</h2>
          {jobs.length === 0 && <p>No jobs yet.</p>}
          <ul style={{ listStyle: "none", padding: 0 }}>
            {jobs.map((job) => (
              <JobRow key={job.id} initialJob={job} />
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
