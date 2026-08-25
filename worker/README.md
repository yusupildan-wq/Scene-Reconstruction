# Legacy worker experiments

The product no longer uses a separate Serverless reconstruction implementation.
Both Local NVIDIA and RunPod pod execution call `scripts/execute_v3_workspace.py`,
which invokes the existing V3 VGGT + gsplat pipeline.

`runner.py` is retained only as historical V2/COLMAP experiment code.
