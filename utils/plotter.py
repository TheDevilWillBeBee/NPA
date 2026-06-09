import numpy as np
from io import BytesIO
from PIL import Image
import torch

from .splatting import splat_particles

@torch.no_grad()
def plot_particles_plt(x, color, title='', lo=-2.1, hi=2.1, figsize=(6,6), sigma=5.0, plot_ax_lines=False):
    import matplotlib.pyplot as plt

    # Convert to numpy for plotting
    points = x.cpu().numpy()
    colors = (color[...,-3:]).clip(0,1).cpu().numpy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.scatter(
        points[:, 0], 
        points[:, 1], 
        cmap='gray',
        vmin=0,
        vmax=1,
        c=colors,
        s=sigma,
        alpha=1.0,
        edgecolor='none'
    )
    
    # Customize the plot
    if not title:
        title = f'Random Points in [-1, 1] Range (N={points.shape[0]})'
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if plot_ax_lines:
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])  # fill the whole figure


    # Handle outputs
#     buf = BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight")
#     buf.seek(0)
#     img = Image.open(buf).convert("RGB")
#     plt.close(fig)   # close to avoid display when returning image
#     return img

    buf = BytesIO()
    fig.savefig(buf, transparent=True, pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    plt.close(fig)   # close to avoid display when returning image
    return img


@torch.no_grad()
def plot_particles_plt_vector(x, color, lo=-2.1, hi=2.1, figsize=(6,6), **scatter_args):
    import matplotlib.pyplot as plt

    # Convert to numpy for plotting
    def tonp(arr):
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return arr
    
    points = tonp(x)
    colors = tonp(color[...,-3:].clip(0,1))
    
    # Create the plot
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.axis('off')
    ax.set_aspect("equal", adjustable="box")
    ax.set_position([0, 0, 1, 1])  # fill the whole figure

    ax.scatter(
        points[:, 0], 
        points[:, 1],
        c=colors,
        **scatter_args,
    )
    
    # plt.tight_layout()
    return fig, ax

@torch.no_grad()
def plot_particles_splat(x, color, img_size=512, sigma=1.0, lo=-2.1, hi=2.1, normalize_color=True, pixel_size=2.0/128.0):
    """
    Efficient parallel Gaussian splatting particle visualization in PyTorch.
    Matches matplotlib.scatter orientation (y increases upward).
    
    Args:
        x: [N, 2] tensor of positions in [-2.1, 2.1].
        color: [N, d] tensor, last 3 channels are RGB in [0,1]
        img_size: output resolution (square).
        sigma: Gaussian std in pixels.
        target_background: whether to also splat global target_pos / target_colors.
    
    Returns:
        PIL.Image in RGB.
    """
    device = x.device
    

    # Foreground
    pos = x
    colors = (color[..., -3:])
    generated_image = splat_particles(x, colors, img_size, sigma, lo, hi, normalize_color=normalize_color, y_up=True, pixel_size=pixel_size)[..., :3].clamp(0.0, 1.0)
    img = generated_image.clamp(0, 1)

    img_uint8 = (img.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(img_uint8)


@torch.no_grad()
def plot_particles_cv(x, color, title='', target_background=True, target_pos=None, target_colors=None, img_size=512, point_size=2, lo=-1.1, hi=1.1):
    import cv2

    """
    Fast point cloud visualization without matplotlib.
    Matches matplotlib.scatter orientation.
    
    Args:
        x: [N, 2] torch tensor (positions in [-2.1, 2.1] range).
        color: [N, d] torch tensor, last 3 channels are RGB in [0,1]
        target_background: whether to draw target_pos, target_colors globals.
        img_size: output image resolution (square).
        point_size: radius of points in pixels.
    
    Returns:
        PIL.Image
    """
    # White background
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8) * 255

    # Normalize to image coords
    def normalize_coords(coords):
        # scale [-2.1, 2.1] -> [0, img_size-1]
        coords = (coords - lo) / (hi - lo) * (img_size - 1)
        # Flip y-axis to match matplotlib (y up instead of down)
        coords[..., 1] = img_size - 1 - coords[..., 1]
        coords = np.clip(coords, 0, img_size - 1)
        return coords.astype(np.int32)

    # Input cloud
    def tonp(arr):
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return arr
    
    points = tonp(x)
    colors = tonp((color[..., -3:]).clip(0, 1))

    # Optionally draw target background
    if target_background:
        tgt_points = tonp(target_pos)
        tgt_colors = tonp(target_colors[...,-3:].clip(0, 1))
        tgt_xy = normalize_coords(tgt_points)
        tgt_col = (tgt_colors * 255).astype(np.uint8)
        for (px, py), col in zip(tgt_xy, tgt_col):
            cv2.circle(img, (px, py), 1, color=tuple(int(c) for c in col), thickness=-1)

    # Draw current points
    xy = normalize_coords(points)
    col = (colors * 255).astype(np.uint8)
    alpha = 0.5 if target_background else 0.9
    for (px, py), c in zip(xy, col):
        c = tuple(int(alpha * v + (1 - alpha) * 255) for v in c)  # blend with white bg
        cv2.circle(img, (px, py), point_size, color=c, thickness=-1)
        

    return Image.fromarray(img)

def plot_particles_splat_batch(x, A, head=4):
    imgs = []
    for b in range(head):
        fig_x, fig_A = x[b], A[b]
        img = plot_particles_splat(fig_x, fig_A + 0.5, img_size=512, lo=-1.2, hi=1.2, sigma=1.0, normalize_color=False)
        imgs.append(np.array(img))
    return Image.fromarray(np.hstack(imgs))


@torch.no_grad()
def plot_vector_field_quiver(
    V: torch.Tensor,
    stride: int = 1,
    eps: float = 1e-8,
    normalize: bool = True,
    scale: float | None = 0.5,
    figsize=(24, 24),
    invert_y: bool = True,   # set False if you want Cartesian (y up)
    dpi=300,
):
    import matplotlib.pyplot as plt

    """
    Save a quiver plot of a 2D vector field V (H, W, 2) to a PNG, skipping near-zero vectors.
    Output: white background, no axes, just arrows.
    """
    assert V.ndim == 3 and V.shape[-1] == 2, f"Expected (H,W,2), got {tuple(V.shape)}"

    Vc = V.detach().to("cpu").float()
    U = Vc[..., 0]
    W = -Vc[..., 1]

    # Subsample
    U = U[::stride, ::stride]
    W = W[::stride, ::stride]

    H, L = U.shape
    y = torch.arange(H)
    x = torch.arange(L)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    # Magnitude + mask
    mag = torch.sqrt(U**2 + W**2)
    keep = mag > eps

    if normalize:
        # Normalize only where we keep vectors
        U = torch.where(keep, U / mag.clamp_min(eps), torch.zeros_like(U))
        W = torch.where(keep, W / mag.clamp_min(eps), torch.zeros_like(W))

    # Flatten and filter to only nonzero vectors
    X = X[keep].numpy()
    Y = Y[keep].numpy()
    U = U[keep].numpy()
    W = W[keep].numpy()

    # transparent background
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    ax.quiver(
        X, Y, U, W,
        angles="xy",
        scale_units="xy",
        scale=scale,
#         width=0.0015,   # tweak for thicker/thinner arrows
    )

    ax.set_aspect("equal", adjustable="box")

    # Force the view to the full (square) grid region
    ax.set_xlim(-0.5, L - 0.5)
    if invert_y:
        ax.set_ylim(H - 0.5, -0.5)
    else:
        ax.set_ylim(-0.5, H - 0.5)

    ax.axis("off")
    ax.set_position([0, 0, 1, 1])  # fill the whole figure

    # IMPORTANT: no bbox_inches="tight" (that causes non-square cropping)
    # Handle outputs
    buf = BytesIO()
    fig.savefig(buf, transparent=True, pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    plt.close(fig)   # close to avoid display when returning image
    return img

    
@torch.no_grad()
def plot_curl_div(
    V: torch.Tensor,
    stride: int = 1,              # subsample for visualization (not for derivatives)
    dx: float = 1.0,
    dy: float = 1.0,
    eps: float = 1e-12,
    robust_norm: bool = True,     # normalize by percentile to avoid outliers dominating
    p: float = 99.0,              # percentile for robust normalization
    figsize=(24, 24),
    invert_y: bool = True,        # match your quiver convention
    dpi: int = 300,
):
    import matplotlib.pyplot as plt

    """
    Computes divergence and scalar curl (2D vorticity) of V and saves an RGBA PNG:
      - curl in RED channel
      - divergence in GREEN channel
      - alpha nonzero where either signal is nonzero (transparent background elsewhere)
    Periodic boundaries are used via torch.roll.

    V: (H, W, 2) with components (u, v) in grid coordinates.
    NOTE: If you flipped Vy for plotting (W = -Vy), that was just a display choice.
          Derivatives here use the actual V as given (u=V[...,0], v=V[...,1]).
    """
    assert V.ndim == 3 and V.shape[-1] == 2, f"Expected (H,W,2), got {tuple(V.shape)}"
    Vc = V.detach().to("cpu").float()

    u = Vc[..., 0]
    v = Vc[..., 1]

    # Central differences with periodic boundary conditions
    du_dx = (torch.roll(u, shifts=-1, dims=1) - torch.roll(u, shifts=1, dims=1)) / (2.0 * dx)
    du_dy = (torch.roll(u, shifts=-1, dims=0) - torch.roll(u, shifts=1, dims=0)) / (2.0 * dy)
    dv_dx = (torch.roll(v, shifts=-1, dims=1) - torch.roll(v, shifts=1, dims=1)) / (2.0 * dx)
    dv_dy = (torch.roll(v, shifts=-1, dims=0) - torch.roll(v, shifts=1, dims=0)) / (2.0 * dy)

    div = du_dx + dv_dy                          # divergence
    curl = dv_dx - du_dy                         # scalar curl (vorticity)

    # Optional subsampling for saved image size
    div = div[::stride, ::stride]
    curl = curl[::stride, ::stride]

    # Robust normalization to [0,1] for visualization (by abs percentile)
    def norm01(x: torch.Tensor) -> torch.Tensor:
        ax = x.abs()
        if robust_norm:
            s = torch.quantile(ax.flatten(), p / 100.0).clamp_min(eps)
        else:
            s = ax.max().clamp_min(eps)
        y = (ax / s).clamp(0.0, 1.0)
        return y

    curl_n = norm01(curl)     # red intensity
    div_n  = norm01(div)      # green intensity

    # Alpha: visible where either is present
    alpha = ((curl.abs() > eps) | (div.abs() > eps)).float()
    # or make alpha proportional to signal strength:
    # alpha = torch.maximum(curl_n, div_n)

    # Build RGBA image
    rgba = torch.zeros((*curl_n.shape, 4), dtype=torch.float32)
    rgba[..., 0] = curl_n      # R
    rgba[..., 1] = div_n       # G
    rgba[..., 2] = 0.0         # B
    rgba[..., 3] = alpha       # A

    rgba_np = rgba.numpy()

    # Plot/save with transparent background, no axes, square canvas
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    ax.imshow(rgba_np, origin="upper", interpolation="nearest")
    ax.set_aspect("equal", adjustable="box")

    if invert_y:
        ax.set_ylim(rgba_np.shape[0] - 0.5, -0.5)
    else:
        ax.set_ylim(-0.5, rgba_np.shape[0] - 0.5)
    ax.set_xlim(-0.5, rgba_np.shape[1] - 0.5)

    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    buf = BytesIO()
    fig.savefig(buf, transparent=True, pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    plt.close(fig)   # close to avoid display when returning image
    return img
