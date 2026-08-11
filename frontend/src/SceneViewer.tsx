import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface GaussianScene {
  means: number[][];
  quats: number[][]; // [w, x, y, z]
  scales: number[][];
  opacities: number[];
  colors: number[][];
}

export default function SceneViewer({ sceneUrl }: { sceneUrl: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [gaussianCount, setGaussianCount] = useState<number | null>(null);

  useEffect(() => {
    let disposed = false;
    let renderer: THREE.WebGLRenderer | undefined;
    let animationFrame: number;

    fetch(sceneUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((data: GaussianScene) => {
        if (disposed || !containerRef.current) return;
        setGaussianCount(data.means.length);

        const container = containerRef.current;
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111);

        const camera = new THREE.PerspectiveCamera(
          50,
          container.clientWidth / container.clientHeight,
          0.01,
          1000
        );

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        // Scene extent is unknown ahead of time (depends on what was trained) --
        // frame the camera from the actual data instead of hardcoding a distance.
        const centroid = new THREE.Vector3();
        data.means.forEach(([x, y, z]) => centroid.add(new THREE.Vector3(x, y, z)));
        centroid.divideScalar(data.means.length);

        let maxDist = 0.1;
        data.means.forEach(([x, y, z]) => {
          maxDist = Math.max(maxDist, new THREE.Vector3(x, y, z).distanceTo(centroid));
        });
        camera.position.set(centroid.x, centroid.y, centroid.z + maxDist * 2.5);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.copy(centroid);
        controls.enableDamping = true;

        // Each Gaussian rendered as a real ellipsoid: its actual learned position,
        // rotation, and per-axis scale -- not flattened into a generic dot. Order-
        // dependent alpha blending (what makes "real" splats look soft/translucent)
        // is skipped for this first version; these render as solid, opaque shapes.
        const geometry = new THREE.SphereGeometry(1, 8, 8);
        const material = new THREE.MeshBasicMaterial();
        const mesh = new THREE.InstancedMesh(geometry, material, data.means.length);

        const matrix = new THREE.Matrix4();
        const position = new THREE.Vector3();
        const quaternion = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        const color = new THREE.Color();

        data.means.forEach((m, i) => {
          position.set(m[0], m[1], m[2]);
          const q = data.quats[i]; // stored as [w, x, y, z]; THREE.Quaternion wants (x, y, z, w)
          quaternion.set(q[1], q[2], q[3], q[0]);
          const s = data.scales[i];
          scale.set(s[0], s[1], s[2]);
          matrix.compose(position, quaternion, scale);
          mesh.setMatrixAt(i, matrix);

          const c = data.colors[i];
          color.setRGB(c[0], c[1], c[2]);
          mesh.setColorAt(i, color);
        });
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        scene.add(mesh);

        const handleResize = () => {
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer?.setSize(container.clientWidth, container.clientHeight);
        };
        window.addEventListener("resize", handleResize);

        const animate = () => {
          controls.update();
          renderer?.render(scene, camera);
          animationFrame = requestAnimationFrame(animate);
        };
        animate();

        return () => {
          window.removeEventListener("resize", handleResize);
          cancelAnimationFrame(animationFrame);
          controls.dispose();
          geometry.dispose();
          material.dispose();
        };
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));

    return () => {
      disposed = true;
      cancelAnimationFrame(animationFrame);
      if (renderer && containerRef.current?.contains(renderer.domElement)) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer?.dispose();
    };
  }, [sceneUrl]);

  return (
    <div>
      {gaussianCount != null && <p>{gaussianCount} Gaussians loaded — drag to orbit, scroll to zoom</p>}
      {error && <p style={{ color: "crimson" }}>Failed to load scene: {error}</p>}
      <div ref={containerRef} style={{ width: "100%", height: 500, background: "#111" }} />
    </div>
  );
}
