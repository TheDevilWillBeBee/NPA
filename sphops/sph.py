import torch
import importlib.util
import os
import sys

_extension_path = os.environ.get("NPA_SPH_EXTENSION_PATH")
if _extension_path:
    # Allow an isolated candidate build to be validated without replacing the
    # installed extension. Importing the module alone does not initialize CUDA.
    sph_cuda = sys.modules.get("sphops.sph_cuda")
    if sph_cuda is not None:
        if os.path.realpath(sph_cuda.__file__) != os.path.realpath(_extension_path):
            raise ImportError("Changing the SPH extension requires a fresh Python process")
    else:
        _spec = importlib.util.spec_from_file_location("sphops.sph_cuda", _extension_path)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Cannot load SPH extension from {_extension_path}")
        sph_cuda = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(sph_cuda)
        sys.modules["sphops.sph_cuda"] = sph_cuda
elif __package__:
    from sphops import sph_cuda
else:
    sph_cuda = sys.modules.get("sphops.sph_cuda")
    if sph_cuda is None:
        import sph_cuda
        sys.modules["sphops.sph_cuda"] = sph_cuda
import math
from dataclasses import dataclass


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


def _fused_perception_channel_count(channels):
    """Smallest native subwarp specialization, or None for unsupported sizes."""
    return next((count for count in (4, 8, 16, 32) if 0 < channels <= count), None)


def fused_perception_supported(x_bin, s_bin, snapshot, mass=None):
    """Fixed-mass, detached-position grid workloads, including padded channels."""
    return (
        x_bin.is_cuda
        and s_bin.is_cuda
        and x_bin.device == s_bin.device
        and not x_bin.requires_grad
        and snapshot.mode == "grid"
        and _fused_perception_channel_count(s_bin.shape[-1]) is not None
        and x_bin.dtype == s_bin.dtype == torch.float32
        and x_bin.is_contiguous()
        and s_bin.is_contiguous()
        and (mass is None or (
            mass.shape in (x_bin.shape[:2], (*x_bin.shape[:2], 1))
            and mass.device == x_bin.device
            and mass.dtype == torch.float32
            and not mass.requires_grad
        ))
        and hasattr(snapshot.grid, "perception_forward")
    )


class FusedPerception(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bin, s_bin, mass, snapshot, eps0, density_grad,
                log_norm_grad, log_norm_density_grad, save_backward):
        B, N, C = s_bin.shape
        dim = snapshot.dim
        volume = s_bin.new_empty(B, N)
        grad_density = s_bin.new_empty(B, N, dim) if density_grad else s_bin.new_empty(0, 0, 0)
        gradient_values = (
            s_bin.new_empty(B, N, C, dim)
            if save_backward and log_norm_grad else s_bin.new_empty(0, 0, 0, 0)
        )
        gradient_sum = s_bin.new_empty(B, N, dim) if save_backward else s_bin.new_empty(0, 0, 0)
        output = s_bin.new_empty(B, N, C * (2 + dim) + (dim if density_grad else 0))
        stream = torch.cuda.current_stream(x_bin.device).cuda_stream
        grid = snapshot.grid
        grid.perception_density_forward(
            x_bin, mass, volume, grad_density, snapshot.bin_offsets,
            snapshot.block_info, density_grad, stream,
        )
        grid.perception_forward(
            x_bin, s_bin, volume, grad_density, eps0, output,
            gradient_values, gradient_sum, snapshot.bin_offsets,
            snapshot.block_info, density_grad, log_norm_grad,
            log_norm_density_grad, save_backward, stream,
        )
        if save_backward:
            ctx.save_for_backward(x_bin, volume, gradient_values, gradient_sum, eps0)
            ctx.snapshot = snapshot
            ctx.state_shape = s_bin.shape
            ctx.density_grad = density_grad
            ctx.log_norm_grad = log_norm_grad
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_bin, volume, gradient_values, gradient_sum, eps0 = ctx.saved_tensors
        snapshot = ctx.snapshot
        B, N, C = ctx.state_shape
        incoming = grad_output.new_empty(B, N, snapshot.dim + 1, C)
        grad_s = grad_output.new_empty(B, N, C)
        snapshot.grid.perception_backward(
            x_bin, volume, grad_output.contiguous(), gradient_values,
            gradient_sum, eps0, incoming, grad_s, snapshot.bin_offsets,
            snapshot.block_info, ctx.density_grad, ctx.log_norm_grad,
            torch.cuda.current_stream(x_bin.device).cuda_stream,
        )
        return None, grad_s, None, None, None, None, None, None, None


def _log_normalize_gradient(value):
    return torch.log(1.0 + value.norm(dim=-1, keepdim=True)) * torch.nn.functional.normalize(value, dim=-1)


def perception(x_bin, s_bin, snapshot, eps0, *, mass=None, state_grad=True,
               density_grad=True, log_norm_grad=True,
               log_norm_density_grad=True, use_fused=True):
    """Pack NPA perception, preserving the hybrid gradient's straight-through backward.

    The fused specialization uses fixed masses and 1..32 channels in grid mode.
    mass is in binned particle order, with shape [B,N] or [B,N,1]; omission
    creates ones, using the same mass-aware kernel. No mass gradients are taken.
    PyTorch zero-pads to native 4/8/16/32 channels and unpacks each feature field;
    both operations preserve autograd. Native sizes need no padding or copies.
    Unsupported shapes, position gradients, and disabled state perception use
    the original differentiable operations. eps0 is a one-element CUDA tensor;
    keeping it on device avoids a scalar synchronization on every rollout step.
    """
    mass = _default_mass(x_bin, mass)
    if mass.shape != x_bin.shape[:2] or mass.device != x_bin.device or mass.dtype != x_bin.dtype:
        raise ValueError("mass must have shape [B,N] or [B,N,1] and match the positions' device and dtype")
    if mass.requires_grad:
        raise ValueError("perception supports fixed masses only; mass must not require gradients")
    mass = mass.contiguous()
    if use_fused and state_grad and fused_perception_supported(x_bin, s_bin, snapshot, mass):
        if eps0.shape != (1,) or eps0.device != x_bin.device or eps0.dtype != torch.float32:
            raise ValueError("eps0 must be a one-element float32 tensor on the input CUDA device")
        if eps0.requires_grad:
            raise ValueError("The fused path does not differentiate the eps0 buffer")
        channels = s_bin.shape[-1]
        padded_channels = _fused_perception_channel_count(channels)
        padded_state = (
            torch.nn.functional.pad(s_bin, (0, padded_channels - channels))
            if padded_channels != channels else s_bin
        )
        save_backward = torch.is_grad_enabled() and padded_state.requires_grad
        output = FusedPerception.apply(
            x_bin, padded_state, mass, snapshot, eps0, density_grad, log_norm_grad,
            log_norm_density_grad, save_backward,
        )
        if padded_channels == channels:
            return output
        # Padding shifts every field after state. A final output truncation
        # would mix channels from different fields and drop density gradients.
        dim = snapshot.dim
        parts = [
            output[..., :channels],
            output[..., padded_channels:padded_channels + channels],
            output[..., 2 * padded_channels:2 * padded_channels + channels * dim],
        ]
        if density_grad:
            parts.append(output[..., padded_channels * (2 + dim):])
        return torch.cat(parts, dim=-1)

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
        grad_rho = grad_rho * scale ** (1 + snapshot.dim) / s_bin.shape[1]
        if log_norm_density_grad:
            grad_rho = _log_normalize_gradient(grad_rho)
        features.append(grad_rho)
    return torch.cat(features, dim=-1)


if __name__ == "__main__":
    # Test code
    from utils import relative_error
    from torch_sph import density as density_torch, blur as blur_torch
    from torch_sph import gradient as gradient_torch, count as count_torch
    from torch_sph import (
        density_gradient as density_gradient_torch,
        moment_matrix as moment_matrix_torch,
    )
    from utils import benchmark_function_forward

    B, N, C = 2, 512, 16
    dim = 2
    eps = 0.1
    sigma = 0.5

    with torch.no_grad():
        x = torch.randn(B, N, dim, device="cuda:0") * sigma
        s = torch.randn(B, N, C, device="cuda:0")
        grid = HashGrid(
            dim=dim,
            boundary="periodic",
            mode="grid",
            grid_size=[16] * dim,
            eps=eps,
            num_particles=N,
            batch_size=B,
        )
        box_size = 16 * eps
        snapshot = grid.instantiate(x)
        x_bin, s_bin = bin_particles(x, s, snapshot)

        co = count(x_bin, snapshot)
        rho = density(x_bin, snapshot)
        s_blur = blur(x_bin, s_bin, rho, snapshot)
        grad_s = gradient(x_bin, s_bin, rho, snapshot)
        density_grad = density_gradient(x_bin, snapshot)
        M = moment_matrix(x_bin, rho, snapshot)

        co_torch = count_torch(x_bin[..., :dim], eps, True, box_size)
        rho_torch = density_torch(x_bin[..., :dim], eps, None, True, box_size)
        s_blur_torch = blur_torch(x_bin[..., :dim], 1.0 / rho_torch, s_bin, eps, None, True, box_size)
        grad_s_torch = gradient_torch(x_bin[..., :dim], 1.0 / rho_torch, s_bin, eps, None, True, box_size)
        density_grad_torch = density_gradient_torch(x_bin[..., :dim], eps, None, True, box_size)
        M_torch = moment_matrix_torch(x_bin[..., :dim], 1.0 / rho_torch, eps, None, True, box_size)

        print("Error Count Forward = ", relative_error(co, co_torch))
        print("Error Density Forward = ", relative_error(rho, rho_torch))
        print("Error Blur Forward = ", relative_error(s_blur, s_blur_torch))
        print("Error Gradient Forward = ", relative_error(grad_s, grad_s_torch))
        print(
            "Error Density Gradient Forward = ",
            relative_error(density_grad, density_grad_torch),
        )
        print("Error Moment Matrix Forward = ", relative_error(M, M_torch))

        # time = benchmark_function_forward(
        #     blur, *(x_bin, s_bin, rho, snapshot), num_runs=512 * 1, warmup_runs=128
        # )
        # print(time)
