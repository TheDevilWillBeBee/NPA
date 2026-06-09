import torch
import numpy as np
import io
import requests
from PIL import Image
from sph import (
    blur,
    density,
    count,
    gradient,
    density_gradient,
    moment_matrix,
    HashGrid,
    bin_particles,
)
from torch_sph import density as density_torch, blur as blur_torch
from torch_sph import gradient as gradient_torch, count as count_torch

from utils import relative_error, benchmark_function_forward


def load_target_image(target_str, size):
    if len(target_str) == 1:
        code_points = [f"{ord(c):04x}" for c in target_str]
        code = "_".join(code_points)
        target_str = (
            "https://github.com/googlefonts/noto-emoji/blob/main/png/512/"
            f"emoji_u{code}.png?raw=true"
        )

    if target_str.startswith(("http://", "https://")):
        response = requests.get(target_str, timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
    else:
        img = Image.open(target_str)

    if isinstance(size, int):
        size = (size, size)

    img = img.convert("RGBA")
    img.thumbnail(size, Image.LANCZOS)
    return np.float32(img) / 255.0


def sample_positions_from_target(
    target_str,
    num_points,
    batch_size,
    device,
    threshold=0.05,
    aabb=(-0.5, 0.5, -0.5, 0.5),
    samples=None,
):
    # Adapt image resolution to keep the number of non-transparent pixels near num_points.
    if samples is None:
        H = 128
        for _ in range(5):
            target_image_np = load_target_image(target_str, size=(H, H))
            nonzero = int((target_image_np[..., 3] >= threshold).sum())
            nonzero = max(nonzero, 1)
            H = max(int(np.round(np.sqrt(num_points / nonzero) * H)), 8)
        samples = [H, H]

    aabb_min = torch.tensor(aabb[0::2], dtype=torch.float32, device=device)
    aabb_max = torch.tensor(aabb[1::2], dtype=torch.float32, device=device)
    aabb_size = aabb_max - aabb_min

    target_image_np = load_target_image(target_str, size=samples)
    target_image = torch.from_numpy(
        target_image_np.transpose(1, 0, 2)[:, ::-1, :].copy()
    ).to(device)

    gshape = torch.tensor(target_image.shape[:2], dtype=torch.float32, device=device)
    grid_x = torch.arange(int(gshape[0].item()), device=device)
    grid_y = torch.arange(int(gshape[1].item()), device=device)
    grid_idx = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="ij"), dim=-1).to(
        torch.float32
    )
    target_pos = aabb_min + aabb_size * (grid_idx + 0.5) / gshape

    target_alpha = target_image[..., 3]
    valid_pos = target_pos[target_alpha > threshold]
    if valid_pos.shape[0] == 0:
        raise ValueError("No valid positions found in target image above alpha threshold.")

    sample_idx = torch.randint(
        0, valid_pos.shape[0], (batch_size, num_points), device=device
    )
    return valid_pos[sample_idx]

B, N, C = 8, 4096, 16
grid_size = 16
dim = 2
eps = 0.1
sigma = 1.0
target_source = "\U0001F98A"
target_threshold = 0.05
target_samples = None
device = torch.device("cuda:0")

with torch.no_grad():
    x = sample_positions_from_target(
        target_str=target_source,
        num_points=N,
        batch_size=B,
        device=device,
        threshold=target_threshold,
        aabb=(-sigma, sigma, -sigma, sigma),
        samples=target_samples,
    )

    if dim != 2:
        # Keep non-2D benchmark shapes compatible with previous behavior.
        extra = torch.randn(B, N, 2, device=device) * sigma * 0.5
        x = torch.cat([x, extra], dim=-1)

    print(x.shape)
    s = torch.randn(B, N, C, device=device)

    # mode = "particle"
    mode = "grid"

    boundary = "clamped"
    # boundary = "periodic"
    
    grid = HashGrid(
        dim=dim,
        boundary=boundary,
        mode=mode,
        grid_size=[grid_size] * dim,
        eps=eps,
        num_particles=N,
        batch_size=B,
        max_particles_per_block=32,
    )
    snapshot = grid.instantiate(x)


loss = 0.0

x = x.clone().detach().requires_grad_(True)
s = s.clone().detach().requires_grad_(True)
x_bin, s_bin = bin_particles(x, s, snapshot)


loss += x_bin.norm() + s_bin.norm()

x_bin = x_bin.clone().detach().requires_grad_(False)
s_bin = s_bin.clone().detach().requires_grad_(True)
co = count(x_bin, snapshot)
rho = density(x_bin, snapshot)
grad_rho = density_gradient(x_bin, snapshot)

loss += rho.norm() + grad_rho.norm()
rho = rho.clone().detach().requires_grad_(False)


s_blur = blur(x_bin, rho, s_bin, snapshot)
grad_s = gradient(x_bin, rho, s_bin, snapshot)
M = moment_matrix(x_bin, rho, snapshot)

loss += s_blur.norm()
loss += grad_s.norm()
loss += M.norm()


loss.backward()
torch.cuda.synchronize()
del x_bin, s_bin, loss, grid, rho, co, s_blur, grad_s, M


# Linux:
# ncu --set full -k regex:".*(forward|backward|permutation|grid_count|block_info).*" --export report4.ncu-rep --force-overwrite --import-source yes --source-folders . python benchmark.py; ncu-ui report4.ncu-rep

# Windows (PowerShell):
# ncu --set full -k 'regex:.*(forward|backward|permutation|grid_count|block_info).*' --export .\report4.ncu-rep --force-overwrite --import-source yes --source-folders . python .\benchmark.py


# Linux:      ncu --set full -k regex:".*(forward|backward|permutation|grid_count|block_info).*" --export ncu_reports/rep1.ncu-rep --force-overwrite --import-source yes --source-folders . python ncu_profile.py
# PowerShell: ncu --set full -k 'regex:".*(forward|backward|permutation|grid_count|block_info).*"' --export ncu_reports/rep1.ncu-rep --force-overwrite --import-source yes --source-folders . python ncu_profile.py