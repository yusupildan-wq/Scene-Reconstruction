"""Real GPU smoke test; an import-only check is deliberately insufficient."""

import torch
from gsplat import rasterization


def main() -> None:
    device = torch.device("cuda")
    means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.05, 0.05, 0.05]], device=device)
    opacities = torch.tensor([0.9], device=device)
    colors = torch.tensor([[1.0, 0.25, 0.1]], device=device)
    viewmats = torch.eye(4, device=device)[None]
    intrinsics = torch.tensor(
        [[[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    rendered, alpha, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        64,
        64,
    )
    torch.cuda.synchronize()
    assert rendered.is_cuda and alpha.is_cuda
    assert torch.isfinite(rendered).all() and float(alpha.max()) > 0
    print(f"gsplat CUDA smoke: healthy ({rendered.shape=}, {float(alpha.max())=:.6f})")


if __name__ == "__main__":
    main()
