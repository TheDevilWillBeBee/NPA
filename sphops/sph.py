import importlib.util
import math
import os
import sys
from dataclasses import dataclass

import torch

_extension_path = os.environ.get("NPA_SPH_EXTENSION_PATH")
if _extension_path:
    # Use an isolated build (e.g. a benchmark candidate) instead of the installed
    # extension. Importing the module alone does not initialize CUDA.
    sph_cuda = sys.modules.get("sphops.sph_cuda")
    if sph_cuda is None:
        _spec = importlib.util.spec_from_file_location("sphops.sph_cuda", _extension_path)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Cannot load SPH extension from {_extension_path}")
        sph_cuda = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(sph_cuda)
        sys.modules["sphops.sph_cuda"] = sph_cuda
    elif os.path.realpath(sph_cuda.__file__) != os.path.realpath(_extension_path):
        raise ImportError("Changing the SPH extension requires a fresh Python process")
else:
    from sphops import sph_cuda


def _default_mass(x, mass):
    if mass is None:
        B, N, _ = x.shape
        return x.new_full((B, N), 1.0)
    if mass.dim() == 3 and mass.shape[-1] == 1:
        return mass.squeeze(-1)
    return mass


def get_hashgrid_cls(dim, boundary, mode):
    return {
        (2, "periodic", "grid"): sph_cuda.HashGrid2DPeriodic,
        (3, "periodic", "grid"): sph_cuda.HashGrid3DPeriodic,
        (2, "clamped", "grid"): sph_cuda.HashGrid2DBox,
        (3, "clamped", "grid"): sph_cuda.HashGrid3DBox,
        (2, "periodic", "particle"): sph_cuda.VHashGrid2DPeriodic,
        (3, "periodic", "particle"): sph_cuda.VHashGrid3DPeriodic,
        (2, "clamped", "particle"): sph_cuda.VHashGrid2DBox,
        (3, "clamped", "particle"): sph_cuda.VHashGrid3DBox,
    }[(dim, boundary, mode)]


@dataclass
class HashGridSnapshot:
    grid: object
    bin_offsets: torch.Tensor
    permutation: torch.Tensor
    block_info: torch.Tensor
    dim: int
    mode: str = "grid"


class HashGrid:
    """
    A class to manage a spatial hash grid for particle simulations.
    This class supports both 2D and 3D grids with periodic or clamped boundary conditions.
    """

    def __init__(
        self,
        dim=2,
        boundary="periodic",
        mode="grid",
        grid_size=(16, 16),
        eps=0.1,
        num_particles=4096,
        batch_size=8,
        max_particles_per_block=64,
    ):
        self.dim = dim
        self.boundary = boundary
        self.mode = mode
        self.grid_size = grid_size
        self.eps = eps
        self.num_particles = num_particles
        self.batch_size = batch_size
        self.max_particles_per_block = max_particles_per_block

        # if mode == "particle" then grid_size must be a power of 2
        if mode == "particle":
            for size in grid_size:
                assert (size & (size - 1)) == 0, "grid_size must be a power of 2"

        self.num_cells = math.prod(grid_size)
        self.hashgrid_cls = get_hashgrid_cls(dim, boundary, mode)
        self.hashgrid = self.hashgrid_cls(
            *grid_size, eps, num_particles, batch_size, max_particles_per_block
        )
        self.device = torch.device("cuda")  # We only support CUDA for now

    def instantiate(self, x):
        """
        Instantiate the hash grid given particle positions and state.

        Args:
            x (torch.Tensor): Particle positions of shape (B, N, dim).
        """
        B, N, _ = x.shape

        permutation = torch.empty(B, N, dtype=torch.int32, device=self.device)
        bin_offsets = torch.empty(
            B * self.num_cells + 1, dtype=torch.int32, device=self.device
        )
        block_info = torch.empty(
            self.hashgrid.max_blocks, 4, dtype=torch.int32, device=self.device
        )

        self.hashgrid.initialize(x, permutation, bin_offsets, block_info)

        return HashGridSnapshot(
            grid=self.hashgrid,
            bin_offsets=bin_offsets,
            permutation=permutation,
            block_info=block_info,
            dim=self.dim,
            mode=self.mode,
        )
    
    def wrap_positions(self, x):
        """
        Wrap particle positions according to the grid's boundary conditions.

        Args:
            x (torch.Tensor): Particle positions of shape (B, N, dim).
        """
        if self.boundary == "periodic":
            x_wrapped = x.clone()
            device = x.device
            x_wrapped = []
            for d in range(self.dim):
                L = torch.tensor(self.grid_size[d] * self.eps, device=device)
                half_L = 0.5 * L
                x_wrapped += [torch.remainder(x[..., d] + half_L, L) - half_L]
            return torch.stack(x_wrapped, dim=-1)
        return x
            


class Permute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, mass, snapshot):
        x_bin = torch.empty_like(x)  # Output
        s_bin = torch.empty_like(s)  # Output
        mass_bin = torch.empty_like(mass)  # Output

        snapshot.grid.bin_particles(
            x, s, mass, snapshot.permutation, x_bin, s_bin, mass_bin
        )
        ctx.snapshot = snapshot
        ctx.x_shape = x.shape
        ctx.s_shape = s.shape
        ctx.m_shape = mass.shape
        ctx.dtype = x.dtype
        ctx.device = x.device
        ctx.dLdx = x.requires_grad
        ctx.dLds = s.requires_grad
        ctx.dLdm = mass.requires_grad
        if not ctx.dLdm:
            # A multi-output Function otherwise marks mass as differentiable
            # whenever state/position requires grad, even for fixed input mass.
            ctx.mark_non_differentiable(mass_bin)
        return x_bin, s_bin, mass_bin

    @staticmethod
    def backward(ctx, *grad_output):
        dL_dx_bin, dL_ds_bin, dL_dm_bin = grad_output
        if dL_dx_bin is None:
            dL_dx_bin = torch.zeros(ctx.x_shape, dtype=ctx.dtype, device=ctx.device)
        if dL_ds_bin is None:
            dL_ds_bin = torch.zeros(ctx.s_shape, dtype=ctx.dtype, device=ctx.device)
        if dL_dm_bin is None:
            dL_dm_bin = torch.zeros(ctx.m_shape, dtype=ctx.dtype, device=ctx.device)

        snapshot = ctx.snapshot
        grad_x = torch.empty_like(dL_dx_bin)  # Output - Gradient w.r.t. input positions
        grad_s = torch.empty_like(dL_ds_bin)  # Output - Gradient w.r.t. input states
        grad_m = torch.empty_like(dL_dm_bin)  # Output - Gradient w.r.t. input mass
        snapshot.grid.backward_bin_particles(
            dL_dx_bin,
            dL_ds_bin,
            dL_dm_bin,
            snapshot.permutation,
            grad_x,
            grad_s,
            grad_m,
        )

        grad_x = grad_x if ctx.dLdx else None
        grad_s = grad_s if ctx.dLds else None
        grad_m = grad_m if ctx.dLdm else None
        return grad_x, grad_s, grad_m, None


def bin_particles(x, s, snapshot, mass=None):
    mass_is_none = mass is None
    mass = _default_mass(x, mass)
    x_bin, s_bin, mass_bin = Permute.apply(x, s, mass, snapshot)
    if mass_is_none:
        return x_bin, s_bin
    return x_bin, s_bin, mass_bin


def unbin_particles(x_bin, s_bin, snapshot, mass_bin=None):
    x = torch.empty_like(x_bin)  # Output
    s = torch.empty_like(s_bin)  # Output
    mass_is_none = mass_bin is None
    mass_bin = _default_mass(x_bin, mass_bin)
    mass = torch.empty_like(mass_bin)
    snapshot.grid.backward_bin_particles(
        x_bin, s_bin, mass_bin, snapshot.permutation, x, s, mass
    )
    if mass_is_none:
        return x, s
    return x, s, mass


class Count(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bin, snapshot):
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info

        count = torch.empty(
            x_bin.shape[:-1], dtype=torch.float32, device=x_bin.device
        )  # Output
        grid.count_forward(x_bin, count, bin_offsets, block_info)
        ctx.save_for_backward(x_bin)
        ctx.snapshot = snapshot
        return count

    @staticmethod
    def backward(ctx, grad_output):
        x_bin = ctx.saved_tensors
        snapshot = ctx.snapshot
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info
        grad_x = torch.empty_like(x_bin)
        # grid.count_backward(x_bin, bin_offsets, grad_output, grad_counts)
        return grad_x, None, None


def count(x_bin, snapshot):
    return Count.apply(x_bin, snapshot)


class Density(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bin, mass, snapshot):
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info

        rho = torch.empty(x_bin.shape[:-1], dtype=torch.float32, device=x_bin.device)
        grid.density_forward(x_bin, mass, rho, bin_offsets, block_info)
        ctx.save_for_backward(x_bin, mass)
        dLdx = x_bin.requires_grad
        ctx.snapshot = snapshot
        ctx.dLdx = dLdx
        return rho

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.dLdx:
            return None, None, None
        x_bin, mass = ctx.saved_tensors
        snapshot = ctx.snapshot
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info

        grad_x = torch.empty_like(x_bin)
        grid.density_backward(
            x_bin, mass, grad_output, grad_x, bin_offsets, block_info, ctx.dLdx
        )
        grad_x = grad_x if ctx.dLdx else None
        return grad_x, None, None


def density(x_bin, snapshot, mass=None):
    mass = _default_mass(x_bin, mass)
    return Density.apply(x_bin, mass, snapshot)


class Blur(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bin, s_bin, rho, mass, snapshot):
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info

        s_blur = torch.empty_like(s_bin)
        grid.blur_forward(x_bin, rho, mass, s_bin, s_blur, bin_offsets, block_info)
        ctx.save_for_backward(x_bin, s_bin, rho, mass)
        ctx.snapshot = snapshot

        ctx.dLdx = x_bin.requires_grad
        ctx.dLds = s_bin.requires_grad
        ctx.dLdrho = rho.requires_grad
        return s_blur

    @staticmethod
    def backward(ctx, grad_output):
        x_bin, s_bin, rho, mass = ctx.saved_tensors
        snapshot = ctx.snapshot
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info

        grad_s = torch.empty_like(s_bin)
        grad_x = torch.empty_like(x_bin)
        grad_rho = torch.empty_like(rho)
        grid.blur_backward(
            x_bin,
            rho,
            mass,
            s_bin,
            grad_output,
            grad_x,
            grad_s,
            grad_rho,
            bin_offsets,
            block_info,
            ctx.dLdx,
            ctx.dLds,
            ctx.dLdrho,
        )
        grad_x = grad_x if ctx.dLdx else None
        grad_s = grad_s if ctx.dLds else None
        grad_rho = grad_rho if ctx.dLdrho else None
        return grad_x, grad_s, grad_rho, None, None


def blur(x_bin, rho, s_bin, snapshot, mass=None):
    mass = _default_mass(x_bin, mass)
    return Blur.apply(x_bin, s_bin, rho, mass, snapshot)


class Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bin, s_bin, rho, mass, snapshot):
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info

        grad_s = torch.empty(
            *s_bin.shape, snapshot.dim, dtype=torch.float32, device=x_bin.device
        )
        grid.gradient_forward(x_bin, rho, mass, s_bin, grad_s, bin_offsets, block_info)
        ctx.save_for_backward(x_bin, s_bin, rho, mass)
        ctx.snapshot = snapshot
        ctx.dLdx = x_bin.requires_grad
        ctx.dLds = s_bin.requires_grad
        ctx.dLdrho = rho.requires_grad
        return grad_s

    @staticmethod
    def backward(ctx, grad_output):
        x_bin, s_bin, rho, mass = ctx.saved_tensors
        snapshot = ctx.snapshot
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info

        grad_s = torch.empty_like(s_bin)
        grad_x = torch.empty_like(x_bin)
        grad_rho = torch.empty_like(rho)
        grid.gradient_backward(
            x_bin,
            rho,
            mass,
            s_bin,
            grad_output,
            grad_x,
            grad_s,
            grad_rho,
            bin_offsets,
            block_info,
            ctx.dLdx,
            ctx.dLds,
            ctx.dLdrho,
        )
        grad_x = grad_x if ctx.dLdx else None
        grad_s = grad_s if ctx.dLds else None
        grad_rho = grad_rho if ctx.dLdrho else None
        return grad_x, grad_s, grad_rho, None, None


def gradient(x_bin, rho, s_bin, snapshot, mass=None):
    mass = _default_mass(x_bin, mass)
    return Gradient.apply(x_bin, s_bin, rho, mass, snapshot)


class DensityGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bin, mass, snapshot):
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info

        grad_out = torch.empty(
            *x_bin.shape[:-1], snapshot.dim, dtype=torch.float32, device=x_bin.device
        )
        grid.density_gradient_forward(x_bin, mass, grad_out, bin_offsets, block_info)
        ctx.save_for_backward(x_bin, mass)
        ctx.snapshot = snapshot
        ctx.dLdx = x_bin.requires_grad
        return grad_out

    @staticmethod
    def backward(ctx, grad_output):
        x_bin, mass = ctx.saved_tensors
        snapshot = ctx.snapshot
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info

        grad_x = torch.empty_like(x_bin)
        grid.density_gradient_backward(
            x_bin, mass, grad_output, grad_x, bin_offsets, block_info, ctx.dLdx
        )
        grad_x = grad_x if ctx.dLdx else None
        return grad_x, None, None


def density_gradient(x_bin, snapshot, mass=None):
    mass = _default_mass(x_bin, mass)
    return DensityGradient.apply(x_bin, mass, snapshot)


class MomentMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bin, rho, mass, snapshot):
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info

        dim = snapshot.dim
        moment_matrix = torch.empty(
            *x_bin.shape[:-1], dim, dim, dtype=torch.float32, device=x_bin.device
        )  # Output [B, N, dim, dim]
        grid.moment_matrix_forward(
            x_bin, rho, mass, moment_matrix, bin_offsets, block_info
        )
        ctx.save_for_backward(x_bin, rho, mass)
        ctx.snapshot = snapshot
        ctx.dLdx = x_bin.requires_grad
        ctx.dLdrho = rho.requires_grad
        return moment_matrix

    @staticmethod
    def backward(ctx, grad_output):
        x_bin, rho, mass = ctx.saved_tensors
        snapshot = ctx.snapshot
        grid = snapshot.grid
        bin_offsets = snapshot.bin_offsets
        block_info = snapshot.block_info
        grad_x = torch.empty_like(x_bin)
        grad_rho = torch.empty_like(rho)
        grid.moment_matrix_backward(
            x_bin,
            rho,
            mass,
            grad_output,
            grad_x,
            grad_rho,
            bin_offsets,
            block_info,
            ctx.dLdx,
            ctx.dLdrho,
        )
        grad_x = grad_x if ctx.dLdx else None
        grad_rho = grad_rho if ctx.dLdrho else None
        return grad_x, grad_rho, None, None


def moment_matrix(x_bin, rho, snapshot, mass=None):
    mass = _default_mass(x_bin, mass)
    return MomentMatrix.apply(x_bin, rho, mass, snapshot)


@torch.no_grad()
def safe_inv_sym_mat(M, rtol=1e-6):
    M_inv = torch.empty_like(M)
    sph_cuda.safe_inv_sym_mat(M, M_inv, rtol)
    return M_inv


def gradient_hybrid(x_bin, rho, s_bin, snapshot, mass=None):
    with torch.no_grad():
        M = moment_matrix(x_bin, rho, snapshot, mass=mass)  # [B, N, dim, dim]
        if snapshot.dim == 2 or snapshot.dim == 3:
            L = safe_inv_sym_mat(M, rtol=1e-3)
        else:
            L = torch.linalg.pinv(M, hermitian=True, rtol=1e-3)

    raw_grad_s = gradient(x_bin, rho, s_bin, snapshot, mass=mass)  # [B, N, C, dim]
    grad_s = raw_grad_s + (raw_grad_s @ L - raw_grad_s).detach()  # [B, N, C, dim]
    return grad_s

    # return raw_grad_s @ L  # [B, N, C, dim]


_NATIVE_CHANNELS = (4, 8, 16, 32)


def _native_channel_count(channels):
    """Smallest fused-kernel channel count holding `channels`; None above 32."""
    return next((count for count in _NATIVE_CHANNELS if 0 < channels <= count), None)


def fused_perception_supported(x, s, grid, mass=None):
    """Whether fused_perception accepts these inputs (otherwise use perception())."""
    return (
        grid.mode == "grid"
        and hasattr(grid.hashgrid, "fused_perception_forward")
        and x.is_cuda
        and s.device == x.device
        and x.dtype == s.dtype == torch.float32
        and _native_channel_count(s.shape[-1]) is not None
        and (mass is None or (
            mass.shape in (x.shape[:2], (*x.shape[:2], 1))
            and mass.device == x.device
            and mass.dtype == torch.float32
            and not mass.requires_grad
        ))
    )


class FusedPerception(torch.autograd.Function):
    """Grid construction, binning, and perception in one native call per direction.

    Inputs are in the original particle order with a native channel count.
    Backward adds the binned-state cotangent to the perception adjoint while
    unbinning, the sum autograd would otherwise form for the two uses of s_bin.
    """

    @staticmethod
    def forward(ctx, x, s, mass, grid, eps0, density_grad, log_norm_grad,
                log_norm_density_grad, save_backward):
        hashgrid = grid.hashgrid
        B, N, C = s.shape
        dim = grid.dim
        snapshot = torch.empty(hashgrid.fused_snapshot_words, dtype=torch.int32, device=x.device)
        x_bin = torch.empty_like(x)
        s_bin = torch.empty_like(s)
        output = s.new_empty(B, N, C * (2 + dim) + (dim if density_grad else 0))
        workspace = s.new_empty(hashgrid.fused_workspace_size(
            C, density_grad, log_norm_grad, save_backward, mass is not None))
        hashgrid.fused_perception_forward(
            x, s, mass, eps0, snapshot, x_bin, s_bin, output, workspace,
            density_grad, log_norm_grad, log_norm_density_grad, save_backward,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        ctx.save_for_backward(x_bin, workspace, eps0, snapshot)
        ctx.grid = grid
        ctx.channels = C
        ctx.density_grad = density_grad
        ctx.log_norm_grad = log_norm_grad
        ctx.has_mass = mass is not None
        ctx.save_backward = save_backward
        if not save_backward:
            ctx.mark_non_differentiable(output)
        return output, x_bin, s_bin, snapshot

    @staticmethod
    def backward(ctx, grad_output, grad_x_bin, grad_s_bin, _):
        x_bin, workspace, eps0, snapshot = ctx.saved_tensors
        if not ctx.save_backward or not ctx.needs_input_grad[1]:
            grad_output = None
        if not ctx.needs_input_grad[1]:
            grad_s_bin = None
        if not ctx.needs_input_grad[0]:
            grad_x_bin = None
        grad_x = torch.empty_like(x_bin) if grad_x_bin is not None else None
        grad_s = (x_bin.new_empty(*x_bin.shape[:2], ctx.channels)
                  if grad_output is not None or grad_s_bin is not None else None)
        if grad_x is None and grad_s is None:
            return (None,) * 9
        # Perception backward scratch: incoming cotangents and the binned state adjoint.
        scratch = (workspace.new_empty(x_bin.shape[0] * x_bin.shape[1] * ctx.channels * (ctx.grid.dim + 2))
                   if grad_output is not None else workspace.new_empty(0))
        ctx.grid.hashgrid.fused_perception_backward(
            x_bin, workspace,
            grad_output.contiguous() if grad_output is not None else None,
            grad_x_bin.contiguous() if grad_x_bin is not None else None,
            grad_s_bin.contiguous() if grad_s_bin is not None else None,
            eps0, snapshot, scratch, grad_x, grad_s, ctx.channels,
            ctx.density_grad, ctx.log_norm_grad, ctx.has_mass,
            torch.cuda.current_stream(x_bin.device).cuda_stream,
        )
        return grad_x, grad_s, None, None, None, None, None, None, None


def fused_perception(x, s, grid, eps0, *, mass=None, perceive_state=True, density_grad=True,
                     log_norm_grad=True, log_norm_density_grad=True):
    """Instantiate the grid, bin, and perceive with the fused CUDA kernels.

    Returns (z, x_bin, s_bin, snapshot). Equivalent, up to float32 summation
    order, to `snapshot = grid.instantiate(x)`, `x_bin, s_bin =
    bin_particles(x, s, snapshot)`, and `z = perception(x_bin.detach(), s_bin,
    snapshot, eps0)`, with s_bin detached inside perception when perceive_state
    is False. Positions get gradients through binning only. Inputs must satisfy
    fused_perception_supported; mass is in the input particle order. States
    with other than 4/8/16/32 channels are zero-padded for the kernels, and
    z and s_bin keep the logical channel count.
    """
    if eps0.shape != (1,) or eps0.device != x.device or eps0.dtype != torch.float32 or eps0.requires_grad:
        raise ValueError("eps0 must be a one-element float32 tensor on the input CUDA device without gradients")
    if mass is not None:
        mass = mass.reshape(x.shape[:2]).contiguous()
    channels = s.shape[-1]
    native = _native_channel_count(channels)
    s_native = s.contiguous()
    if native != channels:
        s_native = torch.nn.functional.pad(s_native, (0, native - channels))
    save_backward = perceive_state and s.requires_grad and torch.is_grad_enabled()
    z, x_bin, s_bin, buffer = FusedPerception.apply(
        x.contiguous(), s_native, mass, grid, eps0, density_grad, log_norm_grad,
        log_norm_density_grad, save_backward,
    )
    if native != channels:
        # Padding shifts every field after state, so slice each field separately.
        dim = grid.dim
        parts = [z[..., :channels], z[..., native:native + channels],
                 z[..., 2 * native:2 * native + channels * dim]]
        if density_grad:
            parts.append(z[..., native * (2 + dim):])
        z = torch.cat(parts, dim=-1)
        s_bin = s_bin[..., :channels]
    B, N = x.shape[:2]
    blocks = grid.hashgrid.max_blocks * 4
    offsets = blocks + B * grid.num_cells + 1
    snapshot = HashGridSnapshot(
        grid=grid.hashgrid, bin_offsets=buffer[blocks:offsets], permutation=buffer[offsets:].view(B, N),
        block_info=buffer[:blocks].view(-1, 4), dim=grid.dim, mode=grid.mode,
    )
    return z, x_bin, s_bin, snapshot


def _log_normalize_gradient(value):
    return torch.log(1.0 + value.norm(dim=-1, keepdim=True)) * torch.nn.functional.normalize(value, dim=-1)


def perception(x_bin, s_bin, snapshot, eps0, *, mass=None, state_grad=True,
               density_grad=True, log_norm_grad=True, log_norm_density_grad=True):
    """NPA perception from the standalone SPH operators on binned particles.

    Packs [state, blur, moment-corrected state gradient, density gradient],
    the gradients optionally log-normalized, with the hybrid gradient's
    straight-through backward. Differentiable with respect to positions and
    states. mass is fixed and in binned particle order ([B,N] or [B,N,1]);
    omitted masses are ones. The density gradient is divided by each sample's
    total mass (N for unit masses). eps0 is a one-element tensor; keeping it on
    the device avoids a host synchronization per step. fused_perception
    computes the same features with the fused kernels.
    """
    unit_mass = mass is None
    mass = _default_mass(x_bin, mass)
    if mass.shape != x_bin.shape[:2] or mass.device != x_bin.device or mass.dtype != x_bin.dtype:
        raise ValueError("mass must have shape [B,N] or [B,N,1] and match the positions' device and dtype")
    if mass.requires_grad:
        raise ValueError("perception supports fixed masses only; mass must not require gradients")
    mass = mass.contiguous()
    rho = density(x_bin, snapshot, mass=mass)
    features = [s_bin, blur(x_bin, rho, s_bin, snapshot, mass=mass)]
    scale = snapshot.grid.eps / eps0
    if state_grad:
        grad_s = gradient_hybrid(x_bin, rho, s_bin, snapshot, mass=mass) * scale
        if log_norm_grad:
            grad_s = _log_normalize_gradient(grad_s)
        features.append(grad_s.flatten(-2))
    if density_grad:
        grad_rho = density_gradient(x_bin, snapshot, mass=mass)
        total_mass = s_bin.shape[1] if unit_mass else mass.sum(dim=1)[:, None, None]
        grad_rho = grad_rho * scale ** (1 + snapshot.dim) / total_mass
        if log_norm_density_grad:
            grad_rho = _log_normalize_gradient(grad_rho)
        features.append(grad_rho)
    return torch.cat(features, dim=-1)
