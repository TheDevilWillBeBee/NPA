import math
import torch
import numpy as np


def splat_particles(
    x,
    color,
    img_size=256,
    sigma=2.0,
    lo=-1.0,
    hi=1.0,
    normalize_color=True,
    y_up=False,
    pixel_size=2.0 / 128.0,
):
    """
    Differentiable Gaussian splatting of particles into an image.

    Args:
        x: (N, 2) positions in world coordinates
        A: (N, D) attributes, last 3 channels are colors in [0,1] (no clamping applied)
        img_size: output image resolution (H=W)
        sigma: Gaussian sigma in pixel units
        lo, hi: world coordinate bounds (map x from [lo,hi] to [0,img_size-1])

    Returns:
        splat: (H, W, 4) tensor with [r, g, b, density] channels
    """
    device = x.device
    dtype = x.dtype
    N = x.shape[0]

    # Map from world coords to pixel coords
    pos_pix = (x - lo) / (hi - lo) * (img_size - 1)
    # Flip y to match matplotlib scatter (y up)
    if y_up:
        pos_pix = torch.stack([pos_pix[:, 0], (img_size - 1) - pos_pix[:, 1]], dim=1)

    # Gaussian kernel radius (local support)

    if normalize_color:
        # convert sigma to pixel space
        sigma = sigma * img_size * pixel_size / (hi - lo)
    radius = max(1, math.ceil(5 * float(sigma)))
    P = 2 * radius + 1

    # Offsets for kernel
    offs = torch.arange(-radius, radius + 1, device=device, dtype=torch.int64)
    yy, xx = torch.meshgrid(offs, offs, indexing="ij")  # [P,P]
    dx = xx.reshape(-1)  # [P2]
    dy = yy.reshape(-1)  # [P2]

    # integer pixel center for each point
    base = torch.floor(pos_pix)  # [N,2]
    cx = base[:, 0].long()
    cy = base[:, 1].long()
    frac = pos_pix - base  # fractional offset inside pixel, [N,2]

    # pixel coordinates of patches
    x_idx = cx[:, None] + dx[None, :]  # [N,P2]
    y_idx = cy[:, None] + dy[None, :]  # [N,P2]

    # valid mask
    mask = (
        (x_idx >= 0) & (x_idx < img_size) & (y_idx >= 0) & (y_idx < img_size)
    )  # [N, P2]

    # distances (subpixel aware)
    dx_f = dx[None, :] - frac[:, 0:1]  # [N,P2]
    dy_f = dy[None, :] - frac[:, 1:2]
    g = torch.exp(-(dx_f**2 + dy_f**2) / (2 * sigma * sigma))  # [N,P2]

    if normalize_color:
        g = g / (1e-8 + torch.sum(g, dim=-1, keepdim=True))
        g = g * (img_size * pixel_size / (hi - lo)) ** 2.0

    # colors (N,3)
    colors = color[..., -3:]

    # flatten indices
    flat_idx = (y_idx * img_size + x_idx)[mask]  # [K=NP2]
    w = g[mask]  # [K]
    c = colors[:, None, :].expand(-1, P * P, -1)[mask, :]  # [K,3]

    if flat_idx.numel() == 0:
        return torch.zeros(img_size, img_size, 4, device=device, dtype=dtype)

    # accumulate into density and weighted colors
    density = torch.zeros(img_size * img_size, 1, device=device, dtype=dtype)
    rgb = torch.zeros(img_size * img_size, 3, device=device, dtype=dtype)

    density.scatter_add_(0, flat_idx[:, None], w[:, None])
    #     print(density.mean())
    rgb.scatter_add_(0, flat_idx[:, None].expand(-1, 3), c * w[:, None])

    # stack [density, rgb]

    out = torch.cat([rgb, density], dim=1).view(img_size, img_size, 4)
    return out


def splat_particles_batch(
    x,
    color,
    img_size=256,
    sigma=2.0,
    lo=-1.0,
    hi=1.0,
    normalize_color=True,
    y_up=False,
    pixel_size=2.0 / 128.0,
):
    """
    Batched splat_particles: (B, N, 2) positions and (B, N, D) attributes to a
    (B, H, W, 4) image, in one pass without a per-sample loop or host sync.

    Per-pixel arithmetic matches splat_particles. Instead of compacting the
    in-image kernel taps with a boolean mask (which synchronizes with the host),
    off-image taps add an exact zero to a clamped in-image pixel.
    """
    B, N, _ = x.shape
    B2, N2, _ = color.shape
    assert B == B2 and N == N2
    device = x.device
    dtype = x.dtype

    pos_pix = (x - lo) / (hi - lo) * (img_size - 1)
    if y_up:
        pos_pix = torch.stack([pos_pix[..., 0], (img_size - 1) - pos_pix[..., 1]], dim=-1)

    if normalize_color:
        sigma = sigma * img_size * pixel_size / (hi - lo)
    radius = max(1, math.ceil(5 * float(sigma)))
    P = 2 * radius + 1
    offs = torch.arange(-radius, radius + 1, device=device, dtype=torch.int64)

    base = torch.floor(pos_pix)  # [B,N,2]
    cx = base[..., 0].long()
    cy = base[..., 1].long()
    frac = pos_pix - base

    # The P x P tap grid is separable: row i has y offset offs[i], column j has
    # x offset offs[j] (splat_particles' meshgrid "ij" order, flattened i*P+j).
    x_idx = cx[..., None] + offs  # [B,N,P]
    y_idx = cy[..., None] + offs
    mask = ((y_idx >= 0) & (y_idx < img_size))[..., :, None] & (
        (x_idx >= 0) & (x_idx < img_size)
    )[..., None, :]  # [B,N,P,P]
    flat_idx = (
        y_idx.clamp(0, img_size - 1)[..., :, None] * img_size
        + x_idx.clamp(0, img_size - 1)[..., None, :]
        + (torch.arange(B, device=device) * img_size * img_size)[:, None, None, None]
    ).reshape(B, N, P * P)
    mask = mask.reshape(B, N, P * P)

    dx_f = offs - frac[..., 0:1]  # [B,N,P]
    dy_f = offs - frac[..., 1:2]
    distance = (dx_f**2)[..., None, :] + (dy_f**2)[..., :, None]  # dx^2 + dy^2
    g = torch.exp(-distance.reshape(B, N, P * P) / (2 * sigma * sigma))

    if normalize_color:
        g = g / (1e-8 + torch.sum(g, dim=-1, keepdim=True))
        g = g * (img_size * pixel_size / (hi - lo)) ** 2.0

    colors = color[..., -3:]
    w = torch.where(mask, g, 0.0)  # [B,N,P2]
    # Zero masked products too, so off-image taps never carry nonfinite colors.
    c = torch.where(mask[..., None], colors[:, :, None, :] * w[..., None], 0.0)

    flat_idx = flat_idx.reshape(-1, 1)
    density = torch.zeros(B * img_size * img_size, 1, device=device, dtype=dtype)
    rgb = torch.zeros(B * img_size * img_size, 3, device=device, dtype=dtype)
    density.scatter_add_(0, flat_idx, w.reshape(-1, 1))
    rgb.scatter_add_(0, flat_idx.expand(-1, 3), c.reshape(-1, 3))

    return torch.cat([rgb, density], dim=1).view(B, img_size, img_size, 4)


def splat_particles_sections(x, color, sections, **kwargs):
    xx = torch.split(x, sections, dim=0)
    cc = torch.split(color, sections, dim=0)
    outs = [splat_particles(bx, bc, **kwargs) for bx, bc in zip(xx, cc)]
    return torch.stack(outs, dim=0)
