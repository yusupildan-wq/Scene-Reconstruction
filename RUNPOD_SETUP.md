# Running the full-room reconstruction on RunPod

Why this exists: `experiments/colab_full_room_pipeline.ipynb` works, but on
Colab's free-tier T4 it's fighting real memory limits at every stage (see
`CLAUDE_HANDOFF.md` -- a 96-frame attempt exhausted memory during DUSt3R
alignment, and `MAX_JOBS` had to be capped to stop gsplat's own CUDA
compilation from exhausting system RAM before training even starts). Colab
sessions also fully reset on disconnect: no persistent disk, so a crash mid-run
means starting over from the video. RunPod fixes both -- a persistent
workspace with a real GPU, and a disk that survives stopping the pod. This
guide is what to set up and what to change; provisioning/paying for it is
your call, I can't do that part.

## 1. Provision a pod

- GPU: an L40S (48GB VRAM) is the specific recommendation in
  `CLAUDE_HANDOFF.md`, based on the diagnosed problem (needing more frames +
  higher resolution than a T4 can hold, not needing more raw compute speed).
  A 24GB card (e.g. RTX 4090) is a cheaper fallback if you want to test the
  pipeline changes below before committing to a bigger instance -- it just
  means staying closer to the current ~64-frame/1024px budget.
- System RAM: at least 64GB. This project has already hit two separate
  system-RAM (not VRAM) exhaustion bugs this session (DUSt3R global alignment
  at 96 frames, gsplat's parallel CUDA compilation) -- undersizing this again
  just relocates the same failure mode to a more expensive machine.
- Template: a PyTorch + CUDA template with Jupyter Lab (RunPod's standard
  "RunPod PyTorch" templates include this). Check RunPod's own pricing page
  for current $/hr -- I don't have live pricing and don't want to quote a
  number that's wrong or stale.
- Storage: attach a persistent volume, not just the pod's ephemeral disk --
  the whole point is surviving a pod stop/restart. Point `CHECKPOINT_DIR` in
  the training cell (and wherever you save frames/pair-graph output, see
  below) at a path on that volume.

## 2. What in the notebook needs to change for RunPod vs. Colab

`colab_full_room_pipeline.ipynb` currently assumes Colab in two places:

- **Video upload** (`from google.colab import files; uploaded = files.upload()`)
  -- `google.colab` doesn't exist outside Colab. On RunPod, either upload the
  video through Jupyter Lab's own file browser (drag-and-drop) and point
  `video_filename` at that path directly, or `scp`/`rsync` it onto the pod's
  persistent volume before starting Jupyter. Replace that cell with:
  ```python
  video_filename = '/workspace/your_video.mov'  # wherever you placed it
  ```
- **Result download** (`files.download(...)`, appears twice: `dust3r_scene.bin`
  and `dust3r_scene_cameras.json`) -- also Colab-only. On RunPod, just leave
  the files where they're written (ideally already on the persistent volume)
  and pull them off with `scp`/`rsync`/RunPod's file browser instead of
  triggering a browser download.
- **`MAX_JOBS=2`, frame cap (64), `TARGET_LONG_EDGE=1024`** -- these were all
  sized for a Colab T4's constraints. On an L40S with real headroom, this is
  exactly the "increase coverage beyond 64 frames... using the larger memory
  budget" step from `CLAUDE_HANDOFF.md` -- raise these deliberately and watch
  memory, not blindly.

Everything else in the notebook (DUSt3R, the diagnostics gates, the
loop-closure pairing, Gaussian training, checkpointing) is plain Python/PyTorch
and runs identically on RunPod.

## 3. Checkpointing across a pod stop

`worker/runner.py`'s `save_training_state`/`load_training_state` (added this
session) serialize the full resumable training state -- Adam momentum, the
Gaussian density-control strategy's accumulated statistics, exposure and
camera-pose-refinement state, not just the current Gaussian values. Point
`CHECKPOINT_DIR` at the persistent volume, and a stopped/restarted pod can
resume exactly where training left off using the "Optional: continue training
from a saved checkpoint" cells in the notebook. **This save/load path is
unvalidated** -- it hasn't been exercised on a real GPU yet (Colab or
RunPod); the first real run is also the first real test of it.

Also worth persisting to the volume, per `CLAUDE_HANDOFF.md`'s "save/checkpoint"
list, beyond what the notebook currently automates: the selected frame
indices, the pair graph (temporal + loop-closure), and the raw DUSt3R
poses/point maps -- so a from-scratch restart doesn't need to re-run DUSt3R
inference, only Gaussian training. The notebook doesn't do this automatically
yet; a straightforward next step would be `np.savez`-ing `pts3d`, `masks`,
`camera_viewmats`, `camera_Ks`, and `frame_paths` right after the diagnostics
gates pass, before Gaussian training starts.

## 4. Before spending real GPU time

Run the pipeline up through the diagnostics cells (frame extraction, DUSt3R,
reprojection gate, adjacent + loop-closure consistency gates) first. All of
that is comparatively cheap. Only start the actual Gaussian training cell once
those gates pass -- that's the entire point of `CLAUDE_HANDOFF.md`'s "run a
bounded geometry test before authorizing another long Gaussian-training run."
