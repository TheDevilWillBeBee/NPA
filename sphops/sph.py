import torch

try:
    from sphops import sph_cuda
except:
    import sph_cuda
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
