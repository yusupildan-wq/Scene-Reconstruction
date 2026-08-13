# Full-room capture baseline (2026-08-13)

Source video: `C:\Users\yusup\Downloads\IMG_1705.mov` (kept outside Git)

- 1920x1080, approximately 30 FPS
- 3,751 source frames / 125.04 seconds
- Covers the bed, bookshelves, window, doors, desk, chair, floor, and repeat passes
- Local preflight produced 175 sharp candidates at a 15-frame sampling interval

Use the updated `extract_frames` default cap of 96 frames for DUSt3R. The cap
is applied uniformly across all sharp candidates so the full camera path is
represented instead of over-weighting whichever area was filmed most slowly.

This capture replaces the earlier desk-heavy video as the next full-room
photorealism baseline. Preserve the 370k checkpoint as the old-scene control.
