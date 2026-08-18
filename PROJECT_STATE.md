# Scene Reconstruction — Project State and Handoff

Last updated: 2026-08-18

This document is the ground-truth handoff for continuing development in a fresh Codex conversation. It combines the current repository state with verified results from the recent Colab and RunPod experiments. It deliberately distinguishes the older 32-camera office/desk experiment from the current 128-view bedroom/full-room experiment. They are different datasets and their metrics must never be compared as though they came from one run.

## 1. Project Purpose

The product goal is a direct pipeline:

`video upload -> camera/geometry reconstruction -> Gaussian training -> quality validation -> browser-viewable, freely navigable photorealistic 3D replica`

The target is not merely a recognizable point cloud or a good still image from one camera. Success means the entire captured room remains coherent, sharp, and photographic while the user moves through it interactively. The desired product eventually exposes this through the application itself: the user uploads a video, the backend schedules GPU work, progress and failures are visible, outputs are retained, and the finished Gaussian scene opens in the browser viewer.

The project currently has all major conceptual pieces, but they are not yet connected into one reliable production workflow. The direct high-quality full-room path is currently run manually on RunPod using Python CLI modules in `experiments/`. The application backend and frontend exist, but the current RunPod experiment is not yet an end-to-end backend job.

User working preference: when introducing new software, machine-learning, or neural-network concepts, explain them thoroughly and in plain language. Give exact commands rather than vague instructions such as “run the cell.” Avoid expensive reruns unless a diagnostic result justifies them.

## 2. Current Architecture

The repository has four practical layers:

1. **Frontend** — React/Vite application and Gaussian scene viewer. It provides project UI and an interactive browser renderer, including camera presets and movement controls.
2. **Backend** — FastAPI application for uploads, projects, jobs, artifacts, storage, and orchestration. This is the intended product control plane.
3. **Worker/training code** — Python implementation of Gaussian initialization, gsplat rasterization, optimization, densification, checkpointing, camera exposure support, optional camera-pose refinement, and export.
4. **Experiments/direct pipeline** — executable Python modules used to reconstruct and train the current full-room dataset on RunPod without relying on a notebook.

The important architectural fact is that the direct RunPod scripts are presently ahead of the product integration. They prove and debug the reconstruction/training workflow, but the backend does not yet automatically create a RunPod job, stream its logs, preserve its checkpoint, and publish its PLY to a project.

The representation is 3D Gaussian Splatting, not a conventional triangle mesh. Each learned Gaussian has a 3D position, anisotropic scale, rotation, opacity, and appearance coefficients. The renderer projects those oriented ellipsoids into screen-space and alpha-composites them. Treating the export as a simple XYZ/RGB point cloud loses essential information and produces fuzzy-dot rendering.

## 3. Important Files

### Current direct full-room path

- `experiments/run_full_room_reconstruction.py` — direct video-to-DUSt3R reconstruction CLI. Extracts frames, constructs the graph, runs inference/global alignment, and checkpoints geometry to disk.
- `experiments/run_full_room_gaussians.py` — validates reconstructed geometry, diagnoses cross-view consistency, trains/resumes Gaussians, evaluates exact training cameras, and exports PLY/NPZ/checkpoints.
- `experiments/reconstruction_graph.py` — graph/profile definitions for temporal connections and reconstruction behavior.
- `experiments/loop_closure_pairs.py` — adds loop-closure pairs so a room traversal can reconnect spatially distant-in-time observations.
- `experiments/test_full_room_gaussians.py` — tests for the direct Gaussian CLI behavior.
- `worker/runner.py` — core Gaussian data structures, rasterization, optimizer, densification, checkpoint state, exposure parameters, optional SH appearance, and optional pose refinement.

### Application

- `backend/app/` — FastAPI application, pipeline/orchestration, dispatch, routers, persistence, and artifact handling.
- `frontend/` — React/Vite user interface.
- `frontend/src/components/GaussianSplatViewer.tsx` — interactive Gaussian viewer and navigation/camera behavior.

### Historical/reference material

- `experiments/colab_dust3r_pipeline.ipynb` — notebook used for earlier experiments. It remains useful as a historical diagnostic record, but new core pipeline work should be direct Python.
- `CLAUDE_HANDOFF.md` — older handoff describing an earlier 64-camera phase. Useful history, but superseded by this document and current code.
- `RUNPOD_SETUP.md` — contains useful RunPod setup concepts, but portions are notebook-era and may be stale.
- `README.md` — currently describes an older/aspirational architecture and still refers to incomplete worker behavior. Do not assume it describes the active experimental pipeline.

### Local untracked evidence

At the time this handoff was created, these pre-existing paths were untracked:

- `experiments/artifacts/2026-08-17-full-room-8000/`
- `experiments/artifacts/full-room-photoreal-pair-graph.json`

Do not delete or overwrite them. They belong to the existing worktree and may contain experiment evidence.

## 4. Current Pipeline

### Reconstruction stage

The current full-room reconstruction module performs the following:

1. Reads the uploaded video.
2. Selects a bounded set of frames according to a reconstruction profile.
3. Loads the images through DUSt3R preprocessing.
4. Creates a sequential/local pair graph plus loop-closure connections.
5. Runs DUSt3R pair inference with batch size 1.
6. Runs `GlobalAlignerMode.PointCloudOptimizer` to jointly align camera poses, focal values, depth maps, and pairwise geometry.
7. Writes durable geometry artifacts, including camera data, per-view point maps, masks, and a completion marker.

For the verified full-room run:

- Input: `/workspace/IMG_1705.mov`
- Input size observed: approximately 227 MB
- Reconstruction output: `/workspace/full-room-photoreal/geometry`
- Views: 128
- Point maps: 128
- Masks: 128
- Camera archive: `cameras.npz`
- Completion marker: `COMPLETE.json`
- DUSt3R image size: 512
- Alignment iterations: 600
- Temporal pair radius: 5, plus loop-closure pairs

The geometry directory is checkpointed before Gaussian training. This is important: a failed or interrupted training experiment must not force another expensive DUSt3R reconstruction.

### Scene loading and validation

`experiments/run_full_room_gaussians.py` loads the saved geometry. Camera poses saved as camera-to-world are inverted to view matrices for rendering. The loader samples dense point maps and associated colors, applies masks, and scales intrinsics to the processed target resolution.

Before training, a hard self-reprojection validation rejects geometry if median error exceeds 2 pixels or p95 error exceeds 10 pixels. The verified run passed with:

- median reprojection: approximately 0.0697 pixels
- p95 reprojection: approximately 0.1797 pixels

These values prove that the stored points, intrinsics, and poses are internally consistent under the validation calculation. They do **not** alone prove that every saved camera corresponds to the correct RGB image or that all conventions/crops match the image presented to the training loss.

### Gaussian training

The trainer initializes Gaussians from reconstructed colored points and optimizes them through gsplat rasterization. The live training state can include optimizer and densification state so continuation is exact rather than a fresh optimization from a PLY.

Baseline profile defaults:

- 3,000 iterations
- target long edge: 1,024
- maximum initial points: 160,000
- densification through step 2,400
- no configured late learning-rate decay
- no configured higher-degree SH
- no weak-view exclusion

Photoreal profile defaults:

- 8,000 iterations
- target long edge: 1,440
- maximum initial points: 400,000
- densification through step 6,000
- learning-rate reduction at step 7,000
- spherical harmonics degree 2
- exclusion of diagnosed weak views

Current direct training enables per-camera exposure optimization and leaves camera-pose optimization disabled. That choice is intentional until the camera/image registration issue is understood; unconstrained pose optimization can hide a pipeline bug or deform the solution.

### Evaluation

Evaluation loads the latest NPZ checkpoint, selects eight approximately evenly spaced cameras, renders with gsplat using the stored view matrices and intrinsics, and writes:

- a JSON metrics report
- `evaluation_latest.jpg`

This is an exact-camera evaluation, not a browser novel-view test. Because current exact-camera renders are already poor, browser engineering is not the active bottleneck for this dataset.

## 5. Environment and Infrastructure

### Local repository

- Repository: `https://github.com/yusupildan-wq/Scene-Reconstruction`
- Branch: `main`
- HEAD when this handoff was prepared: `e525106` (`Add high-detail full-room photoreal training profile`)
- Local branch matched `origin/main` when inspected.

### RunPod

Verified environment from the recent run:

- GPU: NVIDIA L40S, 48 GB VRAM
- Driver: 570.133.20
- CUDA reported by driver: 12.8
- Container family: RunPod PyTorch 2.8
- Python observed in package paths: 3.12
- Persistent network volume: `unchanged_gray_kite_volume`
- Volume size/location: 50 GB, EU-NL-1
- Persistent mount: `/workspace`

RunPod’s CUDA-capable driver and PyTorch container provide the base GPU environment. gsplat still builds/loads its CUDA extension for the active GPU. The first build took approximately 341 seconds. `MAX_JOBS=2` is used to limit parallel compilation because unconstrained compilation previously contributed to host-RAM exhaustion.

The pod’s assigned GPU is not reserved while the pod is stopped. An L40S can be allocated to another user, requiring a migration or a different available machine. The network volume is the durable asset; container-local state and packages may need to be recreated after migration.

Stop the pod when GPU work is finished to stop compute charges. Do not terminate a pod or delete a volume until critical outputs have been downloaded or verified on the persistent volume.

### Python import path

Commands use:

```bash
PYTHONPATH=/workspace/dust3r:/workspace/Scene-Reconstruction:/workspace/Scene-Reconstruction/worker
```

This makes the DUSt3R checkout, project package, and legacy top-level `runner` import available. Missing this path caused earlier `ModuleNotFoundError: No module named 'runner'` failures.

Common runtime packages used in the ad hoc pod included:

```bash
pip install -q gsplat scipy requests pillow
```

This is not yet a sufficiently reproducible deployment mechanism. The production path should pin dependencies in a container or lock file rather than installing them manually for every pod.

## 6. Photorealism Work

### What worked on the older desk scene

The older office/desk dataset contained 32 cameras and approximately 370,000 Gaussians. It achieved good exact-camera reproduction:

- canonical mean: approximately 30.95 dB PSNR, 0.9414 SSIM
- exposure-adjusted mean: approximately 31.02 dB PSNR, 0.9413 SSIM
- refined-pose mean reported in a separate test: approximately 31.34 dB PSNR, 0.9432 SSIM
- a single-view overfit reached 51.37 dB PSNR and 0.9978 SSIM

The single-view result is especially important. It verified that the rasterizer, differentiable loss, and optimizer can reproduce an image almost perfectly when cross-view consistency is removed. Therefore the core renderer is capable of sharp output; the difficult part is constructing one geometry/appearance field that agrees with many cameras.

The old desk experiment also verified finite camera gradients and small useful pose corrections. Its final pose changes were tiny: median rotation around 0.024 degrees and median translation around 0.000257. Those findings apply to that dataset, not automatically to the new bedroom run.

### Current full-room work

The new full-room run expanded coverage to 128 views, added loop closures, persisted aligned geometry, created direct Python training/evaluation scripts, and used an L40S so larger experiments could run without Colab’s session/RAM restrictions.

Cross-view depth diagnostic results:

- median adjacent/local pair inlier ratio: `0.9535889958310493`
- median pair relative error: `0.00487880501896143`
- weak views: `125`, `126`

The photoreal profile excluded those weak views, raised the render resolution, increased initialization density, continued densification longer, enabled degree-2 SH, and trained for 8,000 steps. This was a disciplined attempt to test whether capacity and coverage were the main limitation.

The result did not become photorealistic. It grew to 897,545 Gaussians and exported a 147.2 MB PLY, but the exact-camera mean was only 16.69 dB PSNR / 0.7121 SSIM. This is decisive evidence that merely increasing resolution, Gaussian count, SH capacity, and training duration is not solving the active problem.

## 7. Failed Approaches / Do Not Repeat Blindly

1. **Do not keep adding training iterations without a diagnostic.** The baseline moved from 3,000 to 8,000 steps with only a small aggregate improvement. The photoreal profile at 8,000 steps was worse than baseline.
2. **Do not assume more Gaussians means sharper output.** Increasing from roughly 484,000 to roughly 898,000 Gaussians did not solve blur and reduced the measured mean PSNR.
3. **Do not restart DUSt3R because a later training command failed.** The aligned geometry is checkpointed and survived pod migration. Reuse it.
4. **Do not optimize only one camera and call it a reconstruction improvement.** The 51.37 dB single-view result is a diagnostic proving capacity, not a usable multi-view scene.
5. **Do not tune the browser renderer while Python exact-camera renders are bad.** The browser may still need parity work later, but it cannot restore detail absent from the trained representation.
6. **Do not rely on notebook state.** Colab runtime restarts erased variables, imports, and installed packages, producing repeated `frame_paths` and `runner` errors. The core path is now direct Python.
7. **Do not use all-pairs DUSt3R indiscriminately for a sequential video.** Earlier complete graphs created excessive work and memory pressure. The current temporal/loop-closure graph is deliberate.
8. **Do not allow unbounded CUDA extension compilation.** Use `MAX_JOBS=2` on current RunPod machines unless memory measurements justify increasing it.
9. **Do not treat a low internal reprojection error as proof of photographic registration.** A convention or indexing mistake can remain internally self-consistent.
10. **Do not perform another expensive run until the exact camera/image association is independently visualized and verified.**

Historical infrastructure failures included Colab GPU quota denial, volatile runtime state, host-RAM crashes, slow fallback CUDA compilation, and missing packages after pod migration. They are operational issues, not evidence about scene quality.

## 8. Verified Findings

- The current full-room geometry artifact contains 128 camera records, 128 point maps, and 128 masks.
- Its self-reprojection validation is excellent numerically: median about 0.0697 px and p95 about 0.1797 px.
- Most neighboring-view depth relationships are internally consistent; views 125 and 126 were clear weak outliers.
- The Gaussian renderer can fit a single view to near-perfect quality on the older dataset.
- The older 32-view desk dataset can reach roughly 31 dB mean exact-camera PSNR.
- The current 128-view full-room dataset cannot: exact-camera means remain roughly 17–18 dB and visually blurry.
- The poor quality appears in the Python gsplat evaluation, before browser export/rendering.
- Baseline continuation from 3,000 to 8,000 steps produced only modest improvement.
- The higher-resolution, higher-density photoreal profile regressed to 16.69 dB mean PSNR.
- Therefore the active bottleneck is not simply insufficient GPU speed, iteration count, or Gaussian count.
- The persistent RunPod volume retained the video, reconstruction geometry, and training checkpoint across pod migration.

## 9. Current Hypotheses

These are hypotheses, not verified facts, ranked by current plausibility.

### Hypothesis 1: camera/image/preprocessing mismatch

The highest-priority possibility is an indexing, convention, crop, resize, or intrinsics mismatch between the DUSt3R outputs and the RGB images used by Gaussian training. Examples include a pose assigned to the wrong frame, camera-to-world versus world-to-camera confusion, image rotation metadata, or an intrinsics update that does not exactly match the crop/resize applied to the training image.

Why it fits the evidence: internal point-to-camera reprojection can remain excellent if the same mistaken convention is used consistently, while image-space supervision is badly misregistered. It also explains why adding model capacity does not recover sharpness.

### Hypothesis 2: dense point aggregation creates contradictory surfaces

The loader samples colored points independently from many dense DUSt3R maps. Duplicate or slightly inconsistent surfaces may initialize thick geometry. Large Gaussians can then average competing structures into blur. Cross-view depth consistency is encouraging but does not prove the combined initialization is clean in world space.

### Hypothesis 3: capture limitations

The source video contains high-contrast windows, possible autofocus/exposure changes, motion blur, rolling shutter, and areas with weak parallax or repeated texture. These can limit any static-scene reconstruction. Exposure optimization cannot correct geometric motion blur or rolling-shutter pose variation.

### Hypothesis 4: optimization/loss configuration

Uniform random view sampling and a primarily pixel-space reconstruction loss may underweight edges, small details, difficult regions, or cameras with different exposure. A better loss, resolution schedule, per-view sampling strategy, opacity/scale regularization, or delayed SH schedule may help after registration is proven correct.

### Hypothesis 5: pose refinement is needed

Small residual pose errors can cause multi-view averaging. However pose refinement must come after independent camera/image verification; otherwise it can compensate for the wrong problem. The older desk run benefited only from very small corrections.

## 10. Current Best Result

There are two meanings of “best,” and they must be kept separate.

### Best demonstrated photorealism

The older 32-camera desk scene remains the best-quality reconstruction, at about 31 dB mean exact-camera PSNR with good visual detail near observed views. It did not provide coherent quality for the full room or for large free-camera movement.

### Best current full-room artifact

The most feature-complete full-room artifact is:

- output directory: `/workspace/full-room-photoreal-v2`
- checkpoint: `/workspace/full-room-photoreal-v2/checkpoints/training_state_latest.pt`
- PLY: `/workspace/full-room-photoreal-v2/gaussians_step8000.ply`
- evaluation montage: `/workspace/full-room-photoreal-v2/evaluation_latest.jpg`
- steps: 8,000
- Gaussians: 897,545
- mean PSNR: 16.6856 dB
- mean SSIM: 0.7121

This artifact has the widest/current high-detail configuration but is **not** photorealistic and is not the highest-scoring full-room run.

The baseline full-room checkpoint performed somewhat better numerically after continuation:

- directory: `/workspace/full-room-photoreal`
- Gaussians: 483,556
- approximate eight-view mean after step 8,000: 18.02 dB PSNR

The user downloaded the latest PLY and evaluation montage after the photoreal-v2 evaluation. Do not assume that download is a final product artifact.

## 11. Current RunPod Workflow

### Verify persistent assets

```bash
ls -lh \
  /workspace/IMG_1705.mov \
  /workspace/full-room-photoreal/geometry/COMPLETE.json \
  /workspace/full-room-photoreal/checkpoints/training_state_latest.pt
```

### Install transient dependencies after a fresh/migrated container

```bash
pip install -q gsplat scipy requests pillow
```

### Reconstruct only if geometry is genuinely absent

```bash
cd /workspace/Scene-Reconstruction
nohup env \
  PYTHONPATH=/workspace/dust3r:/workspace/Scene-Reconstruction \
  MAX_JOBS=2 \
  python -u -m experiments.run_full_room_reconstruction \
  /workspace/IMG_1705.mov \
  --profile photoreal \
  --output /workspace/full-room-photoreal \
  > /workspace/full-room-photoreal.log 2>&1 &
tail -f /workspace/full-room-photoreal.log
```

### Validate existing geometry

```bash
cd /workspace/Scene-Reconstruction
PYTHONPATH=/workspace/dust3r:/workspace/Scene-Reconstruction:/workspace/Scene-Reconstruction/worker \
python -u -m experiments.run_full_room_gaussians \
  /workspace/full-room-photoreal \
  --validate-only
```

### Run cross-view depth diagnostics

```bash
cd /workspace/Scene-Reconstruction
PYTHONPATH=/workspace/dust3r:/workspace/Scene-Reconstruction:/workspace/Scene-Reconstruction/worker \
python -u -m experiments.run_full_room_gaussians \
  /workspace/full-room-photoreal \
  --diagnose-cross-view
```

### Train the photoreal profile into a separate directory

```bash
cd /workspace/Scene-Reconstruction
PYTHONPATH=/workspace/dust3r:/workspace/Scene-Reconstruction:/workspace/Scene-Reconstruction/worker \
MAX_JOBS=2 \
python -u -m experiments.run_full_room_gaussians \
  /workspace/full-room-photoreal \
  --profile photoreal \
  --output-dir /workspace/full-room-photoreal-v2
```

### Evaluate without training

```bash
cd /workspace/Scene-Reconstruction
PYTHONPATH=/workspace/dust3r:/workspace/Scene-Reconstruction:/workspace/Scene-Reconstruction/worker \
python -u -m experiments.run_full_room_gaussians \
  /workspace/full-room-photoreal \
  --profile photoreal \
  --output-dir /workspace/full-room-photoreal-v2 \
  --evaluate-only
```

For long runs, use `nohup`, redirect to a log, then `tail -f` the log. `Ctrl+C` while tailing stops only `tail`; it does not stop the background training process. Confirm the training process and completion files separately before stopping the pod.

## 12. Current Work in Progress

The immediate work is diagnosis, not another long training run.

The repository already added:

- a verified full-room reconstruction graph
- durable geometry checkpointing
- bounded direct training
- resumable training state
- direct exact-camera evaluation
- cross-view geometry diagnostics
- a high-detail photoreal profile

The next missing diagnostic is an independent camera/image registration audit. It should produce a human-readable montage per selected view showing:

1. the exact RGB training image
2. projected DUSt3R points colored from that same image
3. a depth visualization
4. frame filename/index
5. pose/intrinsics identifiers and dimensions
6. projected landmarks or edges over the original RGB

This should not reuse the same assumptions invisibly. It needs explicit assertions that frame order, filenames, pose rows, point-map rows, masks, crop boxes, resized dimensions, and intrinsics all refer to the same view.

## 13. Next Steps

### Priority 1 — build an independent registration audit

Add a direct Python diagnostic to inspect at least views around the weakest evaluated regions: 0, 17/18, 35/36, 53/54, 71/72, 89/90, 107/108, and 125–127. Save images and a JSON report. Test it locally with synthetic metadata and on RunPod with the saved geometry. Do not train during this step.

Explicitly verify:

- sorted frame filename order is identical at extraction, DUSt3R inference, geometry save, scene load, and evaluation
- EXIF/video rotation is applied exactly once
- camera-to-world versus world-to-camera conventions
- row-major/column-major matrix handling
- pixel-center convention
- focal/principal-point units
- crop rectangle and resize scale in both axes
- original versus processed image dimensions
- masks align with the final training RGB tensor

### Priority 2 — visualize combined initialization

Render or inspect the initial Gaussian/point configuration before optimization. Quantify duplicate surface thickness and nearest-neighbor disagreement across views. Compare one view’s points alone against the merged 126-view cloud. If the single-view initialization is sharp but the merged initialization is thick, fix fusion rather than optimizer settings.

### Priority 3 — controlled ablations

Only after registration is verified, run short bounded experiments that isolate one variable:

- 8 or 16 contiguous views versus all 126
- one room segment versus full traversal
- original RGB versus normalized exposure
- fixed poses versus tightly regularized pose refinement
- lower versus higher initial point density
- robust/edge-aware loss versus current loss

Each experiment must use the same held-out exact-camera evaluation set and write a named report. Stop failed branches early.

### Priority 4 — capture guidance

If software registration is correct but quality stays low, improve the capture protocol: slow motion, locked focus/exposure/white balance, high shutter speed, multiple overlapping passes, deliberate parallax, and avoidance of moving objects. Bright windows may require a separately exposed pass or locked exposure compromise.

### Priority 5 — product integration

Once a reproducible profile reaches the quality gate, move the direct Python modules behind the backend:

- upload video through the application
- create a persistent reconstruction job
- dispatch to a pinned RunPod image or worker
- stream structured progress
- resume from checkpoints
- save evaluation metrics and montage
- publish the PLY/NPZ and camera metadata
- open the result in the browser viewer
- terminate GPU compute automatically after durable upload

The notebooks can then remain only as optional research examples rather than core application logic.

## 14. Important Design Decisions

- **Direct Python over notebooks:** notebooks are useful for inspection, but production and repeatable experiments must be CLI/module based.
- **Persist reconstruction separately from training:** expensive DUSt3R alignment should be reusable across many Gaussian experiments.
- **Checkpoint optimizer and strategy state:** continuation should be exact; a PLY alone is not sufficient for optimizer continuation.
- **Use a temporal graph plus loop closures:** sequential video structure should be exploited without the memory cost of a complete pair graph.
- **Apply quality gates before expensive training:** invalid geometry must fail early.
- **Evaluate exact cameras before browser views:** this isolates training/export from browser-renderer problems.
- **Keep experiment outputs separate:** baseline and photoreal-v2 directories must not overwrite each other.
- **Bound resource usage:** batch size 1 and `MAX_JOBS=2` are deliberate stability choices.
- **Do not claim photorealism from PSNR alone:** inspect images and novel-view coherence.
- **Preserve the Gaussian representation:** browser export must retain position, anisotropic scale, rotation, opacity, and appearance coefficients.

## 15. Open Questions

1. Are RGB frames, point maps, masks, intrinsics, and poses indexed identically after all sorting and filtering?
2. Does video rotation metadata alter the decoded frame orientation relative to DUSt3R camera coordinates?
3. Does `_crop_region` exactly reproduce the crop chosen by DUSt3R’s `load_images` for every aspect ratio?
4. Are principal points transformed correctly after crop and nonuniform resize?
5. Is the pose inversion correct for both gsplat and the stored DUSt3R convention?
6. How thick is the merged surface initialization compared with each source point map?
7. Are the poor views correlated with graph transitions, low parallax, exposure changes, motion blur, or loop closures?
8. Would a contiguous local subscene train sharply with the current full-room data?
9. Is the evaluation image tensor exactly the tensor used during training, including color space and orientation?
10. After Python quality is fixed, does the browser reproduce an exact camera pixel-for-pixel closely enough?
11. What quantitative gate defines product-ready photorealism across both observed and novel views?

## 16. Project History / Evolution

1. The project began with an application scaffold and an older COLMAP-oriented concept.
2. Experiments moved to DUSt3R for camera/depth reconstruction and gsplat for differentiable Gaussian rendering.
3. The 32-view office/desk experiment progressed from very blurry renders to roughly 31 dB exact-camera quality through screen-space densification, longer training, exposure handling, SH tests, and small pose refinement.
4. Browser viewing exposed large blurry splats and poor novel views. Exact-camera comparisons were introduced to separate viewer problems from scene problems.
5. The goal expanded from one good desk region to a whole-room reconstruction.
6. A new 64-view and then 128-view bedroom/full-room capture was processed. Colab became unreliable because of GPU quotas, volatile state, long reruns, and RAM crashes.
7. Direct Python RunPod scripts replaced the notebook as the active full-room execution path.
8. Reconstruction graph, loop closures, memory bounds, disk checkpointing, direct training, exact-camera evaluation, cross-view diagnostics, and a photoreal profile were added in successive commits.
9. The 128-view geometry passed strong internal validation, but Gaussian exact-camera rendering remained blurry.
10. A high-resolution 8,000-step, ~898k-Gaussian experiment regressed rather than achieving photorealism.
11. The project is now at a diagnostic bottleneck: verify the image/camera/preprocessing contract before spending more GPU time.

## 17. Handoff Warnings

- Do not confuse the old office/desk screenshots and metrics with the current bedroom/full-room run.
- Do not tell the user photorealism has been achieved. It has not.
- Do not recommend another 8,000–16,000-step run as the first action.
- Do not rerun reconstruction unless `/workspace/full-room-photoreal/geometry/COMPLETE.json` or its required artifacts are actually missing/corrupt.
- Do not delete the RunPod network volume or overwrite the current checkpoints.
- Do not assume a stopped RunPod pod retains access to the same L40S.
- Do not use “Run all” notebook instructions as the long-term workflow.
- Do not silently modify unrelated local artifacts or the user’s dirty worktree.
- Do not stage, commit, or push this handoff unless the user explicitly asks.
- Re-check current code and `git status` before acting; repository state may change after this handoff.
- The next agent should lead with evidence and exact commands, teach new ML/software concepts clearly, and minimize unnecessary user-operated steps.

### Immediate recommended action

Implement and test the independent camera/image registration audit described in Section 13. The result of that audit determines the next branch:

- If association, transforms, crop, or intrinsics are wrong, fix that contract and rerun a short bounded training test.
- If they are correct but merged initialization is thick, fix multi-view fusion/point initialization.
- If both are correct, run controlled loss/pose/capture ablations.
- Only after Python exact-camera quality becomes strong again should browser parity and final product automation become the primary focus.
