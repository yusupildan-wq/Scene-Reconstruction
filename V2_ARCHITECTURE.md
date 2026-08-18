# V2 reconstruction architecture

video -> `ns-process-data` (COLMAP-based camera/geometry) -> `splatfacto` (MCMC densification) -> exported .ply + renders

## What's replacing what, and why

- **COLMAP replaces DUSt3R** for camera/geometry. DUSt3R is architecturally a
  pairwise method with a lightweight global-alignment step glued on top.
  COLMAP is a native many-view joint system, built for exactly the "hundreds
  of images of one room" regime -- which is precisely where V1's own
  16-run experiment found the failure (geometric disagreement getting worse
  as view count grew, DUSt3R's weakest regime, not COLMAP's).
- **Nerfstudio's `splatfacto` replaces our custom trainer.** Mature,
  actively maintained, tested across many real-world captures by many users
  -- not something only this project exercises. Built on the same `gsplat`
  library V1 already used, so this isn't throwing away everything, just the
  parts that were custom and undertested.
- **MCMC densification** (a real strategy inside `gsplat`, selectable in
  `splatfacto`) replaces the original split/clone/prune heuristic V1 used.
  Fixed Gaussian budget with stochastic resampling instead of fragile
  gradient-threshold heuristics -- generally considered the more robust of
  the two, and it's a flag, not new code.
- **Frame selection**: using `ns-process-data`'s own default video handling
  for the first run, not V1's custom blur/redundancy filter -- keeping this
  simple until/unless results suggest frame selection specifically is the
  weak link.

## What's NOT changing

Output is still a standard `.ply` Gaussian scene -- same representation V1
produced, same thing the existing `frontend/src/GaussianSplatViewer.tsx`
(`@mkkellogg/gaussian-splats-3d`) already knows how to render. If V2 wins,
the browser side doesn't need to change, only what produces the file.

## Isolation

Everything writes to `/workspace/v2-reconstruction/` on the RunPod volume --
a new directory, not touching `/workspace/full-room-photoreal*` or any other
existing V1/Codex output. This worktree/branch (`claude/v2-reconstruction`)
is separate from `main`, `codex/*`, and `claude/reference-baseline`.
