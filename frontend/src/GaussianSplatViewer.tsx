import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

const MIN_FOV = 20;
const MAX_FOV = 90;

// Optional companion file a Colab export can produce alongside the .ply --
// real camera-to-world matrices from training, so the viewer can start
// exactly where a real photo was taken instead of guessing a position.
// Older/other exports won't have this file; 404 is expected and handled.
async function loadCameraPoses(plyUrl: string): Promise<THREE.Matrix4[] | null> {
  const url = plyUrl.replace(/\.ply$/, "_cameras.json");
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = (await res.json()) as { camera_to_world_matrices: number[][] };
    return data.camera_to_world_matrices.map((flat) => new THREE.Matrix4().fromArray(flat));
  } catch {
    return null;
  }
}

// Renders real oriented-ellipsoid Gaussian Splats (via a proven third-party
// renderer) instead of SceneViewer.tsx's flat circular point sprites -- takes
// a .ply file (the standard 3D Gaussian Splatting format), not our own
// custom .bin/.json formats.
//
// Camera is a first-person walkthrough (PointerLockControls), not an orbit
// around the scene from outside: you start standing at the room's center,
// drag (once clicked/locked) turns your head, WASD walks, scroll zooms the
// lens (FOV) rather than moving you -- avoids "zooming" walking you into a
// wall. DUSt3R doesn't canonicalize which axis is "up", so the starting
// position is an approximation of standing in the room, not a true
// eye-level placement on a known floor.
export default function GaussianSplatViewer({ sceneUrl }: { sceneUrl: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [isLocked, setIsLocked] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;

    let disposed = false;
    let animationFrame: number;
    let cleanupScene: (() => void) | undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.01, 1000);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new PointerLockControls(camera, renderer.domElement);
    const handleLock = () => setIsLocked(true);
    const handleUnlock = () => setIsLocked(false);
    controls.addEventListener("lock", handleLock);
    controls.addEventListener("unlock", handleUnlock);

    const handleClick = () => {
      if (!controls.isLocked) controls.lock();
    };
    container.addEventListener("click", handleClick);

    const keys = { forward: false, backward: false, left: false, right: false };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "KeyW") keys.forward = true;
      if (e.code === "KeyS") keys.backward = true;
      if (e.code === "KeyA") keys.left = true;
      if (e.code === "KeyD") keys.right = true;
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === "KeyW") keys.forward = false;
      if (e.code === "KeyS") keys.backward = false;
      if (e.code === "KeyA") keys.left = false;
      if (e.code === "KeyD") keys.right = false;
    };
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      camera.fov = THREE.MathUtils.clamp(camera.fov + e.deltaY * 0.02, MIN_FOV, MAX_FOV);
      camera.updateProjectionMatrix();
    };
    container.addEventListener("wheel", handleWheel, { passive: false });

    // dropInMode (forced on by DropInViewer's constructor) turns off this
    // library's own render loop/camera/controls, so it plugs into ours
    // instead of owning them. sharedMemoryForWorkers:false avoids needing
    // COOP/COEP response headers on the dev server just to unlock
    // SharedArrayBuffer for a single scene. gpuAcceleratedSort:false is
    // required -- with it enabled, the library's GPU-based splat-distance
    // precomputation (run through our externally-provided renderer/camera,
    // an integration path far less exercised than its own standalone Viewer)
    // silently produces a scene that never actually renders anything, with
    // no error anywhere: correct data, correct uniforms, correct compiled
    // shaders, just zero visible splats. The CPU sort has no such issue.
    const splatViewer = new GaussianSplats3D.DropInViewer({
      gpuAcceleratedSort: false,
      sharedMemoryForWorkers: false,
    });
    scene.add(splatViewer);

    let moveSpeed = 1; // units/sec, rescaled below once the real scene size is known

    // NOTE: addSplatScenes picks its parser from the URL's file extension --
    // a cache-busting query string (e.g. "?t=...") breaks that check and
    // throws synchronously ("File format not supported"), outside this
    // promise chain, so don't add one here.
    Promise.all([splatViewer.addSplatScenes([{ path: sceneUrl }], false), loadCameraPoses(sceneUrl)])
      .then(([, cameraPoses]) => {
        if (disposed) return;
        setLoaded(true);

        // .ply positions aren't pre-centered around the origin the way our
        // own .bin/.json exports were -- start from the real loaded bounds
        // instead of assuming a fixed position.
        //
        // THREE.Box3().setFromObject() reads the "position" vertex attribute
        // of the mesh's geometry, but this library renders every splat as an
        // instance of one small template quad -- each splat's real world
        // position lives in a separate per-instance data buffer, not that
        // attribute, so setFromObject() was measuring the template quad
        // (roughly unit-sized) instead of the actual scene extent. splatMesh
        // .computeBoundingBox() reads the real per-splat centers instead.
        const box = splatViewer.splatMesh.computeBoundingBox();
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();

        if (cameraPoses && cameraPoses.length > 0) {
          // Real photo vantage point: guaranteed to be open, photographed
          // space, not guessed geometry.
          const pose = cameraPoses[Math.floor(cameraPoses.length / 2)];
          const position = new THREE.Vector3();
          const quaternion = new THREE.Quaternion();
          const scale = new THREE.Vector3();
          pose.decompose(position, quaternion, scale);
          camera.position.copy(position);
          camera.quaternion.copy(quaternion);
        } else {
          // No pose data for this scene (older export) -- the raw centroid
          // of every Gaussian's position can land inside a wall or dense
          // clutter for a room scanned from around it, which renders as
          // chaotic noise from point-blank range. Pull back from it instead
          // of standing exactly on it.
          camera.position.set(center.x, center.y, center.z + size * 0.5);
        }
        moveSpeed = Math.max(size / 6, 0.01); // cross the room in ~6 seconds of holding W
      })
      .catch((err: unknown) => {
        // React StrictMode double-invokes this effect in dev (mount ->
        // cleanup -> mount again); the first instance's cleanup disposes
        // its viewer mid-load, which rejects this promise with "Scene
        // disposed" even though the second, real instance loads fine
        // underneath. Only surface rejections from the still-live instance.
        if (disposed) return;
        setError(err instanceof Error ? err.message : String(err));
      });

    const handleResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", handleResize);

    let lastTime = performance.now();
    const animate = () => {
      const now = performance.now();
      const delta = (now - lastTime) / 1000;
      lastTime = now;

      if (controls.isLocked) {
        const step = moveSpeed * delta;
        if (keys.forward) controls.moveForward(step);
        if (keys.backward) controls.moveForward(-step);
        if (keys.right) controls.moveRight(step);
        if (keys.left) controls.moveRight(-step);
      }

      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    cleanupScene = () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      container.removeEventListener("click", handleClick);
      container.removeEventListener("wheel", handleWheel);
      controls.removeEventListener("lock", handleLock);
      controls.removeEventListener("unlock", handleUnlock);
      controls.dispose();
      // DropInViewer has no dispose() of its own -- the underlying Viewer
      // instance (which owns the splat mesh and its sort worker) does.
      splatViewer.viewer.dispose();
    };

    return () => {
      disposed = true;
      cancelAnimationFrame(animationFrame);
      cleanupScene?.();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.forceContextLoss();
      renderer.dispose();
    };
  }, [sceneUrl]);

  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <div
        ref={containerRef}
        style={{ width: "100%", height: "100%", background: "#111", cursor: isLocked ? "none" : "pointer" }}
      />
      {loaded && !isLocked && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(0,0,0,0.35)",
            color: "#fff",
            fontFamily: "system-ui, sans-serif",
            fontSize: 20,
            pointerEvents: "none",
          }}
        >
          Click to look around — WASD to move, scroll to zoom
        </div>
      )}
      {error && (
        <p
          style={{
            position: "absolute",
            top: 12,
            left: 12,
            color: "crimson",
            background: "rgba(0,0,0,0.6)",
            padding: 8,
            borderRadius: 4,
            fontFamily: "system-ui, sans-serif",
          }}
        >
          Failed to load scene: {error}
        </p>
      )}
    </div>
  );
}
