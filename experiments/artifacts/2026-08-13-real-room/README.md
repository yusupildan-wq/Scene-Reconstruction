# Real-room reconstruction checkpoint — 2026-08-13

This folder records the first reconstruction produced with the validated
screen-space gsplat strategy and staged learning-rate schedule.

Quality label: **HIGH QUALITY**, not photorealistic.

Key measured results:

- 370,042 Gaussians
- Five-view fixed-pose canonical result before pose refinement:
  approximately 30.95 dB PSNR / 0.9414 SSIM
- Five-view refined-pose result: 31.34 dB PSNR / 0.9432 SSIM
- Single-view overfit: 51.37 dB PSNR / 0.9978 SSIM
- Three adjacent-view fit: 33.81 / 36.43 / 28.54 dB
- Distant views 0 and 31 fit together: 41.51 / 48.84 dB
- Degree-1 spherical harmonics test regressed to 30.46 dB / 0.9385 SSIM
- Exposure compensation reduced diagnostic adjacent-view RGB MAE from 0.0230
  to 0.0147; canonical reconstruction improvement was modest
- Camera-pose refinement corrections were small: median 0.024390 degrees and
  0.00025653 scene units

The large binary checkpoint and executed notebook are intentionally ignored by
Git. Their expected local filenames in this folder are:

- `best_scene_370k.npz`
- `best_scene_camera_state.npz`
- `colab_dust3r_pipeline_executed.ipynb`

The originals remain in the user's Downloads folder.

SHA-256 checksums:

- `best_scene_370k.npz`:
  `B51B19D698727034EA9700CF872015CE43D198B27DEC3B40655E0FFF16C1FEDA`
- `colab_dust3r_pipeline_executed.ipynb`:
  `4B17C6660A8497953D1AC19BAD804C2250131892694C01C315A49CE8D0DFFD30`
- `best_scene_camera_state.npz`:
  `BA425AEF59D649F13A7B2841FD4989ED5FBB5F34AA0C51320D6C40C9FEEFB9A2`

Camera-state contents:

- `camera_pose_deltas`: `(32, 6)` float32; rotation-vector XYZ followed by
  translation XYZ, applied as a left-multiplied world-to-camera correction
- `exposure_log_gains`: `(32, 3)` float32
- `exposure_biases`: `(32, 3)` float32
