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
type CameraPoseSidecar = {
  camera_to_world_matrices?: number[][];
  frames?: Array<{ transform_matrix?: number[][] | number[] }>;
  scene_scale_factor?: number;
  coordinate_convention?: "opencv" | "opengl";
  world_up?: "y" | "z";
};

type LoadedCameraPoses = {
  poses: THREE.Matrix4[];
  worldRotation: THREE.Quaternion;
};

async function loadCameraPoses(plyUrl: string): Promise<LoadedCameraPoses | null> {
  const url = plyUrl.replace(/\.ply$/, "_cameras.json");
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = (await res.json()) as CameraPoseSidecar;
    const sceneScale = data.scene_scale_factor ?? 1;
    const matrices = data.camera_to_world_matrices ?? data.frames?.flatMap((frame) => {
      if (!frame.transform_matrix) return [];
      const matrix = frame.transform_matrix;
      return [Array.isArray(matrix[0]) ? (matrix as number[][]).flat() : (matrix as number[])];
    });
    if (!matrices?.length) return null;
    const poses = matrices.map((flat) => {
      // Existing project sidecars are column-major. Nerfstudio-style nested
      // transform_matrix values are row-major and were flattened above.
      const isNerfstudioFrame = data.camera_to_world_matrices == null;
      const pose = new THREE.Matrix4();
      if (isNerfstudioFrame) pose.fromArray(flat).transpose();
      else pose.fromArray(flat);
      // convert_bin_to_ply.py uniformly scales Gaussian positions to avoid
      // tiny covariance values in the browser renderer. Camera translations
      // must receive the same scale or the exported photographed viewpoint is
      // no longer in the same coordinate system as the rendered scene.
      pose.elements[12] *= sceneScale;
      pose.elements[13] *= sceneScale;
      pose.elements[14] *= sceneScale;
      // gsplat trains with OpenCV cameras (+Y down, +Z forward), while
      // Three.js cameras use +Y up and look down -Z.
      if (data.coordinate_convention === "opencv") {
        pose.multiply(new THREE.Matrix4().makeScale(1, -1, -1));
      }
      return pose;
    });

    // PCA-normalized reconstructions do not have a guaranteed world-up axis.
    // A phone is held approximately upright, so the robust average of the
    // photographed cameras' local +Y axes gives us the room's actual up.
    const averageUp = poses.reduce((sum, pose) => {
      const elements = pose.elements;
      return sum.add(new THREE.Vector3(elements[4], elements[5], elements[6]).normalize());
    }, new THREE.Vector3()).normalize();
    const worldRotation = new THREE.Quaternion().setFromUnitVectors(averageUp, new THREE.Vector3(0, 1, 0));
    const worldMatrix = new THREE.Matrix4().makeRotationFromQuaternion(worldRotation);
    poses.forEach((pose) => pose.premultiply(worldMatrix));
    return { poses, worldRotation };
  } catch {
    return null;
  }
}

function removeCameraRoll(camera: THREE.PerspectiveCamera) {
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  camera.up.set(0, 1, 0);
  camera.lookAt(camera.position.clone().add(forward));
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
  // All V3 quality profiles share VGGT/gsplat's coordinate convention.
  const isV3 = /\/v3_scene(?:_[^/]+)?\.ply(?:$|\?)/.test(sceneUrl);
  const containerRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const cameraPosesRef = useRef<THREE.Matrix4[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [isLocked, setIsLocked] = useState(false);
  const [activeView, setActiveView] = useState<number | null>(null);
  const [viewOptions, setViewOptions] = useState<number[]>([]);
  const [hasCameraPoses, setHasCameraPoses] = useState(false);

  const jumpToTrainingView = (viewIndex: number) => {
    const camera = cameraRef.current;
    const pose = cameraPosesRef.current[viewIndex];
    if (!camera || !pose) return;
    pose.decompose(camera.position, camera.quaternion, new THREE.Vector3());
    if (isV3) removeCameraRoll(camera);
    camera.updateMatrixWorld(true);
    setActiveView(viewIndex);
  };

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;

    let disposed = false;
    let animationFrame: number;
    let cleanupScene: (() => void) | undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.01, 1000);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new PointerLockControls(camera, renderer.domElement);
    // This scene's imported camera basis makes PointerLockControls feel
    // horizontally reversed; invert look input while retaining corrected WASD.
    controls.pointerSpeed = isV3 ? 1 : -1;
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
      .then(([, cameraData]) => {
        if (disposed) return;
        setLoaded(true);

        if (cameraData && isV3) {
          splatViewer.quaternion.copy(cameraData.worldRotation);
          splatViewer.updateMatrixWorld(true);
        }
        const cameraPoses = cameraData?.poses ?? null;

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
          cameraPosesRef.current = cameraPoses;
          setHasCameraPoses(true);
          const lastView = cameraPoses.length - 1;
          setViewOptions(
            Array.from(new Set([0, 0.25, 0.5, 0.75, 1].map((fraction) => Math.round(lastView * fraction))))
          );
          // Real photo vantage point: guaranteed to be open, photographed
          // space, not guessed geometry.
          const initialView = Math.floor((cameraPoses.length - 1) / 2);
          const pose = cameraPoses[initialView];
          const position = new THREE.Vector3();
          const quaternion = new THREE.Quaternion();
          const scale = new THREE.Vector3();
          pose.decompose(position, quaternion, scale);
          camera.position.copy(position);
          camera.quaternion.copy(quaternion);
          if (isV3) removeCameraRoll(camera);
          setActiveView(initialView);
        } else {
          setHasCameraPoses(false);
          setActiveView(null);
          setViewOptions([]);
          // No pose data for this scene (older export) -- the raw centroid
          // of every Gaussian's position can land inside a wall or dense
          // clutter for a room scanned from around it, which renders as
          // chaotic noise from point-blank range. Pull back from it instead
          // of standing exactly on it.
          camera.position.set(center.x, center.y, center.z + size * 0.5);
          camera.lookAt(center);
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
        // The imported training-camera convention faces opposite Three.js's
        // default local forward axis, so compensate here to preserve normal
        // first-person controls: W advances and S retreats.
        if (keys.forward) controls.moveForward(isV3 ? step : -step);
        if (keys.backward) controls.moveForward(isV3 ? -step : step);
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
      cameraRef.current = null;
      cameraPosesRef.current = [];
    };
  }, [sceneUrl, isV3]);

  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <div
        ref={containerRef}
        style={{ width: "100%", height: "100%", background: "#111", cursor: isLocked ? "none" : "pointer" }}
      />
      {loaded && hasCameraPoses && (
        <div
          onClick={(event) => event.stopPropagation()}
          style={{
            position: "absolute",
            bottom: 16,
            left: 16,
            display: "flex",
            gap: 6,
            padding: 8,
            borderRadius: 8,
            background: "rgba(0,0,0,0.72)",
            color: "white",
            fontFamily: "system-ui, sans-serif",
          }}
        >
          <span style={{ alignSelf: "center", marginRight: 4 }}>Training camera:</span>
          {viewOptions.map((viewIndex) => (
            <button
              key={viewIndex}
              type="button"
              onClick={() => jumpToTrainingView(viewIndex)}
              style={{
                padding: "5px 9px",
                border: activeView === viewIndex ? "1px solid #fff" : "1px solid #777",
                borderRadius: 5,
                background: activeView === viewIndex ? "#2563eb" : "#222",
                color: "white",
                cursor: "pointer",
              }}
            >
              {viewIndex}
            </button>
          ))}
        </div>
      )}
      {loaded && !hasCameraPoses && (
        <div
          style={{
            position: "absolute",
            bottom: 16,
            left: 16,
            padding: "8px 10px",
            borderRadius: 8,
            background: "rgba(120,53,15,0.88)",
            color: "white",
            fontFamily: "system-ui, sans-serif",
            fontSize: 14,
          }}
        >
          Overview only — no training-camera metadata was exported
        </div>
      )}
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
          Click to look — mouse turns • W/S forward/back • A/D left/right • scroll zooms
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
