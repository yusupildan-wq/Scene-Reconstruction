import { useEffect, useState } from "react";
import { createJob, createProject, listJobs, listProjects, Job, Project } from "./api/client";
import JobRow from "./JobRow";
import SceneViewer from "./SceneViewer";

export default function App() {
  const [activeTab, setActiveTab] = useState<"projects" | "demo">("demo");
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

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>Scene Reconstruction</h1>

      <div style={{ marginBottom: 24 }}>
        <button onClick={() => setActiveTab("demo")} disabled={activeTab === "demo"}>
          Demo Scene
        </button>{" "}
        <button onClick={() => setActiveTab("projects")} disabled={activeTab === "projects"}>
          Projects
        </button>
      </div>

      {activeTab === "demo" && (
        <section style={{ marginBottom: 32 }}>
          <h2>Trained Gaussian Splat scene (Colab, 1000 iterations)</h2>
          <SceneViewer sceneUrl="/demo_scene.json" />
        </section>
      )}

      {activeTab === "projects" && (
        <>
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
        </>
      )}
    </div>
  );
}
