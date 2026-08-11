import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface FlatGaussianScene {
  n: number;
  means: Float32Array; // length n*3
  scales: Float32Array; // length n*3
  opacities: Float32Array; // length n
  colors: Float32Array; // length n*3
  // quats intentionally omitted -- unused by this renderer (see note below)
}

interface JsonGaussianScene {
  means: number[][];
  quats: number[][];
  scales: number[][];
  opacities: number[];
  colors: number[][];
}

// Compact binary format (written by the Colab export cell for scenes too large
// for JSON to be practical -- 2M+ Gaussians as JSON text runs into the hundreds
// of MB and is slow to parse; this is a few bytes of overhead plus raw float32,
// exactly what a Float32Array can read directly with zero parsing cost):
//   [uint32 count][count*3 float32 means][count*4 float32 quats]
//   [count*3 float32 scales][count float32 opacities][count*3 float32 colors]
async function loadBinaryScene(url: string): Promise<FlatGaussianScene> {
  const buffer = await fetch(url).then((res) => {
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.arrayBuffer();
  });
  const view = new DataView(buffer);
  const n = view.getUint32(0, true); // little-endian
  let offset = 4;
  const means = new Float32Array(buffer, offset, n * 3);
  offset += n * 3 * 4;
  offset += n * 4 * 4; // skip quats -- unused by this renderer
  const scales = new Float32Array(buffer, offset, n * 3);
  offset += n * 3 * 4;
  const opacities = new Float32Array(buffer, offset, n);
  offset += n * 4;
  const colors = new Float32Array(buffer, offset, n * 3);
  return { n, means, scales, opacities, colors };
}

async function loadJsonScene(url: string): Promise<FlatGaussianScene> {
  const data: JsonGaussianScene = await fetch(url).then((res) => {
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  });
  const n = data.means.length;
  const means = new Float32Array(n * 3);
  const scales = new Float32Array(n * 3);
  const opacities = new Float32Array(n);
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    means[i * 3] = data.means[i][0];
    means[i * 3 + 1] = data.means[i][1];
    means[i * 3 + 2] = data.means[i][2];
    const s = data.scales[i];
    scales[i * 3] = s[0];
    scales[i * 3 + 1] = s[1];
    scales[i * 3 + 2] = s[2];
    opacities[i] = data.opacities[i];
    const c = data.colors[i];
    colors[i * 3] = c[0];
    colors[i * 3 + 1] = c[1];
    colors[i * 3 + 2] = c[2];
  }
  return { n, means, scales, opacities, colors };
}

// Custom shader for THREE.Points: gl_PointCoord gives us a position within each
// point sprite (0-1 on both axes), which lets the fragment shader fade out toward
// the edge (a soft circular falloff) instead of a hard-edged shape -- this is what
// real splats need and a plain THREE.PointsMaterial can't do on its own. Each
// point also gets its own size/alpha/color via per-vertex attributes, since stock
// PointsMaterial only supports one uniform size for every point.
const VERTEX_SHADER = `
  attribute vec3 color;
  attribute float pointSize;
  attribute float pointAlpha;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vColor = color;
    vAlpha = pointAlpha;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = pointSize * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const FRAGMENT_SHADER = `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 fromCenter = gl_PointCoord - vec2(0.5);
    float distSq = dot(fromCenter, fromCenter);
    if (distSq > 0.25) discard; // outside the circular sprite -- fully transparent
    float falloff = exp(-8.0 * distSq); // soft Gaussian-like fade toward the edge
    gl_FragColor = vec4(vColor, vAlpha * falloff);
  }
`;

export default function SceneViewer({ sceneUrl }: { sceneUrl: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [gaussianCount, setGaussianCount] = useState<number | null>(null);

  useEffect(() => {
    let disposed = false;
    let renderer: THREE.WebGLRenderer | undefined;
    let animationFrame: number;

    const load = sceneUrl.endsWith(".bin") ? loadBinaryScene(sceneUrl) : loadJsonScene(sceneUrl);

    load
      .then((data) => {
        if (disposed || !containerRef.current) return;
        const n = data.n;
        setGaussianCount(n);

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

        const centroid = new THREE.Vector3();
        for (let i = 0; i < n; i++) {
          centroid.x += data.means[i * 3];
          centroid.y += data.means[i * 3 + 1];
          centroid.z += data.means[i * 3 + 2];
        }
        centroid.divideScalar(n);

        let maxDist = 0.1;
        const p = new THREE.Vector3();
        for (let i = 0; i < n; i++) {
          p.set(data.means[i * 3], data.means[i * 3 + 1], data.means[i * 3 + 2]);
          maxDist = Math.max(maxDist, p.distanceTo(centroid));
        }
        camera.position.set(centroid.x, centroid.y, centroid.z + maxDist * 2.5);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.copy(centroid);
        controls.enableDamping = true;

        // Each Gaussian's real per-axis scale (x,y,z) is averaged into one size --
        // a point sprite is inherently a circle, not an oriented ellipsoid, so
        // rotation isn't used by this renderer (and isn't loaded at all from the
        // binary format, to save memory/bandwidth on million-point scenes).
        const positions = data.means; // reuse directly, no copy needed
        const colors = data.colors;
        const sizes = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          sizes[i] = ((data.scales[i * 3] + data.scales[i * 3 + 1] + data.scales[i * 3 + 2]) / 3) * 400;
        }
        const alphas = data.opacities;

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions.slice(), 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colors.slice(), 3));
        geometry.setAttribute("pointSize", new THREE.BufferAttribute(sizes, 1));
        geometry.setAttribute("pointAlpha", new THREE.BufferAttribute(alphas.slice(), 1));

        const material = new THREE.ShaderMaterial({
          vertexShader: VERTEX_SHADER,
          fragmentShader: FRAGMENT_SHADER,
          vertexColors: true,
          transparent: true,
          depthWrite: false, // required for correct blending of overlapping
          // transparent points -- but this only looks right if points are drawn
          // back-to-front, which WebGL doesn't do automatically (see below).
        });

        const points = new THREE.Points(geometry, material);
        scene.add(points);

        // WebGL draws points in whatever order they're in the buffer -- with
        // transparency and no depth write, that gives wrong results unless we
        // draw farthest-from-camera first. At small scales (a few thousand
        // points) this was cheap to redo every single frame; at hundreds of
        // thousands of points a full JS sort every frame is real, likely
        // noticeable overhead -- throttled below so it still updates smoothly
        // as you orbit, just not literally 60 times a second for huge scenes.
        const order = new Uint32Array(n);
        for (let i = 0; i < n; i++) order[i] = i;
        const distances = new Float32Array(n);
        const sortedPositions = new Float32Array(n * 3);
        const sortedColors = new Float32Array(n * 3);
        const sortedSizes = new Float32Array(n);
        const sortedAlphas = new Float32Array(n);
        const sortEveryNFrames = n > 200_000 ? 5 : 1;
        let frameCount = 0;

        function sortPointsByDistanceFromCamera() {
          for (let i = 0; i < n; i++) {
            const dx = positions[i * 3] - camera.position.x;
            const dy = positions[i * 3 + 1] - camera.position.y;
            const dz = positions[i * 3 + 2] - camera.position.z;
            distances[i] = dx * dx + dy * dy + dz * dz;
          }
          const orderArray = Array.from(order);
          orderArray.sort((a, b) => distances[b] - distances[a]); // farthest first
          for (let i = 0; i < n; i++) {
            const src = orderArray[i];
            sortedPositions[i * 3] = positions[src * 3];
            sortedPositions[i * 3 + 1] = positions[src * 3 + 1];
            sortedPositions[i * 3 + 2] = positions[src * 3 + 2];
            sortedColors[i * 3] = colors[src * 3];
            sortedColors[i * 3 + 1] = colors[src * 3 + 1];
            sortedColors[i * 3 + 2] = colors[src * 3 + 2];
            sortedSizes[i] = sizes[src];
            sortedAlphas[i] = alphas[src];
          }
          geometry.attributes.position.array.set(sortedPositions);
          geometry.attributes.color.array.set(sortedColors);
          geometry.attributes.pointSize.array.set(sortedSizes);
          geometry.attributes.pointAlpha.array.set(sortedAlphas);
          geometry.attributes.position.needsUpdate = true;
          geometry.attributes.color.needsUpdate = true;
          geometry.attributes.pointSize.needsUpdate = true;
          geometry.attributes.pointAlpha.needsUpdate = true;
        }

        const handleResize = () => {
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer?.setSize(container.clientWidth, container.clientHeight);
        };
        window.addEventListener("resize", handleResize);

        const animate = () => {
          controls.update();
          if (frameCount % sortEveryNFrames === 0) sortPointsByDistanceFromCamera();
          frameCount++;
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
