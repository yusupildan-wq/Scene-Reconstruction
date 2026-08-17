# Active Development Handoff: Full-Room Photorealistic Reconstruction

## Primary objective

Continue this existing project; do not restart it. The target is a freely
navigable, browser-rendered, photorealistic 1:1 replica of the entire captured
room. The user wants decisive implementation and concise instructions. When a
new software, ML, or neural-network concept is introduced, explain it clearly
and thoroughly.

## Current result and firm diagnosis

The latest full-room model is somewhat better than the 3,000-step checkpoint,
but it is still blurry and not photorealistic. This is no longer primarily a
browser-renderer problem or a lack-of-training-iterations problem. Python
renders at the original training cameras are themselves blurry.

Full-room observed-view metrics:

| Checkpoint | Mean PSNR | Mean SSIM |
|---|---:|---:|
| 3,000 steps | 16.44 dB | 0.6603 |
| 8,000 steps | 17.60 dB | 0.7108 |

At 8,000 steps, views 0/15/31/47/63 were respectively:

- 21.30 dB / 0.7540
- 18.41 dB / 0.7710
- 18.76 dB / 0.7025
- 12.92 dB / 0.5634
- 16.59 dB / 0.7631

Five thousand additional steps gained only 1.16 dB on average, and view 47
worsened. Gaussian count stayed at 379,806 because densification had already
stopped at step 2,400. Do not blindly extend this reconstruction to 16,000
steps. The dominant bottleneck is inconsistent camera/geometry recovery across
the room, especially weak cross-room connections and loop closure.

## What has been implemented

- Full-room capture is `IMG_1705.mov`: 1920x1080, about 125 seconds and 3,751
  source frames. It covers the bed, shelves, window, doors, desk, chair, and
  floor with repeated passes.
- `backend/app/pipeline.py` selects up to 64 sharp, distributed frames. A
  96-frame attempt exceeded Colab/T4 memory during DUSt3R alignment.
- Clean notebook: `experiments/colab_full_room_pipeline.ipynb`.
- Initial point count is capped at 160k; DUSt3R objects are released before
  Gaussian training; `MAX_JOBS=2` prevents gsplat CUDA compilation from
  exhausting system RAM.
- Runner imports use a robust absolute path.
- Training supports exposure optimization, late pose refinement, learning-rate
  decay, and exact continuation through live optimizer/strategy state.
- Browser uses `@mkkellogg/gaussian-splats-3d`, an oriented Gaussian PLY—not a
  point-cloud approximation—and loads all 64 exported camera poses.
- Current browser assets have been replaced with the 8,000-step export and the
  frontend production build passes.

Important commits:

- `ffd174d` Prevent gsplat build from exhausting runtime RAM
- `0798142` Bound full-room training memory before quality gate
- `837fc06` Fit full-room alignment within T4 memory
- `2931809` Add clean full-room Colab notebook
- `fd5234f` Prepare full-room photorealism reconstruction

## Preserved artifacts

- `experiments/artifacts/2026-08-17-full-room-3000/`
- `experiments/artifacts/2026-08-17-full-room-8000/`

The 8,000-step archive contains the 379,806-Gaussian binary, 64-camera JSON,
and converted PLY. The active viewer files are:

- `frontend/public/dust3r_scene.bin`
- `frontend/public/dust3r_scene.ply`
- `frontend/public/dust3r_scene_cameras.json`

There are intentional uncommitted changes and large local artifacts. Inspect
them and preserve them; do not reset or overwrite them.

## Correct next path toward photorealism

Work on reconstruction quality before further Gaussian optimization:

1. Move the heavy reconstruction run to a persistent RunPod workspace, ideally
   an L40S-class 48 GB GPU with at least 64 GB system RAM. Persist each stage so
   a failure never requires starting from video again.
2. Save/checkpoint: extracted frames and their indices, pair graph, recovered
   cameras/intrinsics, aligned point maps, filtered initialization, and Gaussian
   optimizer/strategy state.
3. Increase coverage beyond 64 frames using the larger memory budget, but use a
   deliberate connectivity graph—not an all-pairs explosion. Combine temporal
   neighbors with visual loop-closure/cross-room pairs so repeated sightings of
   the desk, window, shelves, and bed constrain one coordinate frame.
4. Add a pre-training cross-view geometry gate. Self-reprojection alone is not
   sufficient because a bad camera and its own bad depth map can agree. Measure
   correspondence/reprojection consistency between independently predicted
   overlapping views, visualize the camera trajectory, identify broken graph
   edges, and refuse expensive Gaussian training when the geometry fails.
5. Investigate a stronger global reconstruction front end (for example a
   modern dense-matching/SfM pipeline) using official implementations and
   evidence. Explain any new model to the user before adopting it. Do not swap
   models merely because they are newer; run a bounded A/B geometry test first.
6. Only after the geometry gate passes, initialize and train Gaussians with
   validation cameras distributed over every room region. Track per-region
   metrics so blank walls cannot inflate the mean and hide failed shelves or
   furniture.
7. Compare Python and browser renders at identical exported cameras. If Python
   becomes sharp but the browser does not match, then debug export/rasterizer
   conventions. At present Python is already blurry, so renderer tuning alone
   cannot solve the main problem.

## Capture limitations to account for

The bright window causes severe exposure variation, and some portions of the
video may have mostly rotational motion with weak translational parallax.
Exposure correction can normalize brightness but cannot invent reliable depth.
If diagnostics show insufficient parallax or unseen surfaces, be honest that a
better capture is required and give an exact capture route rather than wasting
hours on training.

## Immediate action

First inspect the current worktree and the full-room notebook/pipeline. Then
implement a persistent RunPod-ready reconstruction workflow with staged
checkpoints and a stronger loop-closed pair graph. Run a bounded geometry test
before authorizing another long Gaussian-training run. The success criterion is
sharp, coherent observed-view Python renders across the whole room—not merely a
higher iteration count or a good-looking desk view.
