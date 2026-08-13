import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

// Renders real oriented-ellipsoid Gaussian Splats (via a proven third-party
// renderer) instead of SceneViewer.tsx's flat circular point sprites -- takes
// a .ply file (the standard 3D Gaussian Splatting format), not our own
// custom .bin/.json formats.
export default function GaussianSplatViewer({ sceneUrl }: { sceneUrl: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;

    let disposed = false;
    let animationFrame: number;
    let cleanupScene: (() => void) | undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.01, 1000);
    camera.position.set(0, 0, 5); // reframed once real scene bounds are known, below

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

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

    // NOTE: addSplatScenes picks its parser from the URL's file extension --
    // a cache-busting query string (e.g. "?t=...") breaks that check and
    // throws synchronously ("File format not supported"), outside this
    // promise chain, so don't add one here.
    splatViewer
      .addSplatScenes([{ path: sceneUrl }], false)
      .then(() => {
        if (disposed) return;
        setLoaded(true);

        // .ply positions aren't pre-centered around the origin the way our
        // own .bin/.json exports were -- frame the camera from the real
        // loaded bounds instead of assuming a fixed position.
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
        const distance = Number.isFinite(size) && size > 0 ? size * 1.2 : 5;
        camera.position.set(center.x, center.y, center.z + distance);
        controls.target.copy(center);
        controls.update();
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

    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    cleanupScene = () => {
      window.removeEventListener("resize", handleResize);
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
    <div>
      {loaded && <p>Gaussian Splat scene loaded (real ellipsoid rendering) — drag to orbit, scroll to zoom</p>}
      {error && <p style={{ color: "crimson" }}>Failed to load scene: {error}</p>}
      <div ref={containerRef} style={{ width: "100%", height: 500, background: "#111" }} />
    </div>
  );
}
