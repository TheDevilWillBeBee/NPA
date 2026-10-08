import torch
import math
from typing import Tuple, Optional


def _default_mass(x: torch.Tensor, mass: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Return particle masses with default 1/N if not provided.

    Args:
        x: [B, N, DIM] particle positions
        mass: Optional[Tensor] of shape [B, N] or [B, N, 1]

    Returns:
        m: [B, N]
    """
    if mass is None:
        B, N, _ = x.shape
        return x.new_full((B, N), 1.0)
    if mass.dim() == 3 and mass.shape[-1] == 1:
        return mass.squeeze(-1)
    return mass


def _pairwise_displacement(
    x: torch.Tensor,
    periodic: bool,
    box_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute pairwise displacement vectors.

    Args:
        x: [B, N, DIM] positions
        periodic: if True, use periodic displacement on torus in [-1, 1]

    Returns:
        r_vec: [B, N, N, DIM] displacement vectors
    """
    x_i = x.unsqueeze(2)  # [B, N, 1, DIM]
    x_j = x.unsqueeze(1)  # [B, 1, N, DIM]
    r_vec = x_j - x_i
    if periodic:
        # Wrap displacement into [-L/2, L/2) on a torus of size L
        if box_size is None:
            L = 2.0
        else:
            L = box_size
        if not torch.is_tensor(L):
            L = x.new_tensor(L)
        half_L = L * 0.5
        r_vec = torch.remainder(r_vec + half_L, L) - half_L
    return r_vec

def bin_particles(x, s, permutation, mass: Optional[torch.Tensor] = None):
    """
    PyTorch reference implementation of bin_particles.
    Uses the INVERSE of snapshot.permutation to permute x and s.
    
    Args:
        x: Particle positions (B, N, D)
        s: Particle features (B, N, C)
        permutation: Grid permutation indices (B, N)
        mass: Optional[Tensor] of shape [B, N] or [B, N, 1]
    
    Returns:
        x_bin: Inverse permuted positions (B, N, D)
        s_bin: Inverse permuted features (B, N, C)
        m_bin (optional): Inverse permuted masses (B, N)
    """
    B, N, D = x.shape
    C = s.shape[2]
#     permutation = snapshot.permutation
    
    # Expand permutation for scattering along dimension 1
    perm_x = permutation.unsqueeze(-1).expand(-1, -1, D).to(torch.long)
    perm_s = permutation.unsqueeze(-1).expand(-1, -1, C).to(torch.long)
    
    # Apply inverse permutation using scatter
    # x_bin[b, permutation[b, i], :] = x[b, i, :]
    x_bin = torch.zeros_like(x)
    s_bin = torch.zeros_like(s)
    
    x_bin.scatter_(1, perm_x, x)
    s_bin.scatter_(1, perm_s, s)

    if mass is None:
        return x_bin, s_bin

    m = _default_mass(x, mass)
    m_bin = torch.zeros_like(m)
    perm_m = permutation.to(torch.long)
    m_bin.scatter_(1, perm_m, m)
    return x_bin, s_bin, m_bin

def smoothing_poly6_normalization(h: float, dim: int) -> float:
    """Poly6 smoothing kernel normalization factor"""
    if dim == 2:
        return 4.0 / (math.pi * (h ** 8.0))
    elif dim == 3:
        return 315.0 / (64.0 * math.pi * (h ** 9.0))
    else:
        raise NotImplementedError(f"Poly6 normalization for dim={dim} not implemented")


def smoothing_poly6_kernel(r_vec: torch.Tensor, h: float) -> torch.Tensor:
    """
    Poly6 smoothing kernel
    
    Args:
        r_vec: [..., DIM] - relative position vectors
        h: float - smoothing length
    
    Returns:
        [...] - kernel values
    """
    r_squared = torch.sum(r_vec ** 2, dim=-1)
    h_squared = h ** 2
    
    # Only compute for particles within smoothing radius
    mask = r_squared < h_squared
    
    result = torch.zeros_like(r_squared)
    h2_minus_r2 = h_squared - r_squared
    h2_minus_r2 = torch.clamp(h2_minus_r2, min=0.0)
    result = torch.where(mask, h2_minus_r2 ** 3, torch.zeros_like(h2_minus_r2))
    
    return result


def gradient_spiky_normalization(h: float, dim: int) -> float:
    """Spiky gradient kernel normalization factor"""
    if dim == 2:
        return 10.0 / (math.pi * (h ** 5.0))
    elif dim == 3:
        return 15.0 / (math.pi * (h ** 6.0))
    else:
        raise NotImplementedError(f"Spiky normalization for dim={dim} not implemented")

def gradient_spiky_kernel(r_vec: torch.Tensor, h: float) -> torch.Tensor:
    """
    Spiky gradient kernel
    
    Args:
        r_vec: [..., DIM] - relative position vectors
        h: float - smoothing length
    
    Returns:
        [..., DIM] - gradient vectors
    """
    eps = 1e-12
    r_squared = torch.sum(r_vec ** 2, dim=-1, keepdim=True)
    
    # Exact distance for masking
    d = torch.sqrt(r_squared)
    # Safe distance for division
    d_safe = torch.sqrt(r_squared + eps)
    
    mask = (d.squeeze(-1) < h) & (d.squeeze(-1) > 0)
    
    h_minus_r = h - d_safe
    magnitude = 3.0 * torch.clamp(h_minus_r, min=0.0) ** 2 / d_safe
    result = magnitude * r_vec
    
    result = torch.where(mask.unsqueeze(-1), result, torch.zeros_like(result))
    
    return result


def count_kernel(r_vec: torch.Tensor, h: float) -> torch.Tensor:
    """
    Count kernel (simple indicator function)
    
    Args:
        r_vec: [..., DIM] - relative position vectors
        h: float - smoothing length
    
    Returns:
        [...] - count values (0 or 1)
    """
    r_squared = torch.sum(r_vec ** 2, dim=-1)
    return (r_squared < h ** 2).float()


def density(
    x: torch.Tensor,
    h: float,
    mass: Optional[torch.Tensor] = None,
    periodic: bool = False,
    box_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute SPH density using brute force pairwise comparison
    
    Args:
        x: [B, N, DIM] - particle positions
        h: float - smoothing length
        mass: Optional[Tensor] of shape [B, N] or [B, N, 1]
        periodic: if True, use periodic displacement on torus in [-1, 1]
    
    Returns:
        rho: [B, N] - particle volumes
    """
    B, N, DIM = x.shape
    
    # Compute normalization
    normalization = smoothing_poly6_normalization(h, DIM)
    
    m = _default_mass(x, mass)

    # Compute all pairwise distances
    r_vec = _pairwise_displacement(x, periodic, box_size)  # [B, N, N, DIM]
    
    # Compute kernel values
    kernel_vals = smoothing_poly6_kernel(r_vec, h)  # [B, N, N]
    
    # Sum over neighbors (mass density)
    m_j = m.unsqueeze(1)  # [B, 1, N]
    rho = torch.sum(kernel_vals * m_j, dim=2)  # [B, N]
    
    return rho * normalization


def volume(
    x: torch.Tensor,
    h: float,
    mass: Optional[torch.Tensor] = None,
    periodic: bool = False,
    box_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute SPH volumes using brute force pairwise comparison
    
    Args:
        x: [B, N, DIM] - particle positions
        h: float - smoothing length
        mass: Optional[Tensor] of shape [B, N] or [B, N, 1]
        periodic: if True, use periodic displacement on torus in [-1, 1]
    
    Returns:
        v: [B, N] - particle volumes
    """
    rho = density(x, h, mass=mass, periodic=periodic, box_size=box_size)
    m = _default_mass(x, mass)

    return m / rho


def gradient(
    x: torch.Tensor,
    rho: torch.Tensor,
    A: torch.Tensor,
    h: float,
    mass: Optional[torch.Tensor] = None,
    periodic: bool = False,
    box_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute SPH gradients using brute force pairwise comparison
    
    Args:
        x: [B, N, DIM] - particle positions
        rho: [B, N] - particle densities
        A: [B, N, F] - input features
        h: float - smoothing length
        mass: Optional[Tensor] of shape [B, N] or [B, N, 1]
        periodic: if True, use periodic displacement on torus in [-1, 1]
    
    Returns:
        GA: [B, N, F, DIM] - feature gradients
    """
    B, N, DIM = x.shape
    F = A.shape[2]
    
    # Convert density to volume (specific volume)
    m = _default_mass(x, mass)
    v = m / rho
    
    # Compute normalization
    normalization = gradient_spiky_normalization(h, DIM)
    
    # Compute all pairwise distances
    r_vec = _pairwise_displacement(x, periodic, box_size)  # [B, N, N, DIM]
    
    # Compute gradient kernel
    grad_kernel = gradient_spiky_kernel(r_vec, h)  # [B, N, N, DIM]
    
    # Volume weighting
    v_j = v.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
    weighted_grad = grad_kernel * v_j  # [B, N, N, DIM]
    
    # Feature differences
    A_i = A.unsqueeze(2)  # [B, N, 1, F]
    A_j = A.unsqueeze(1)  # [B, 1, N, F]
    dA = A_j - A_i  # [B, N, N, F]
    
    # Compute gradients
    # GA[b, i, f, d] = sum_j (A_j[f] - A_i[f]) * v_j * grad_kernel[b, i, j, d]
    gradients = torch.sum(dA.unsqueeze(-1) * weighted_grad.unsqueeze(3), dim=2)  # [B, N, F, DIM]
    
    return normalization * gradients


def count(
    x: torch.Tensor,
    h: float,
    periodic: bool = False,
    box_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Count neighbors using brute force pairwise comparison
    
    Args:
        x: [B, N, DIM] - particle positions
        h: float - smoothing length
        periodic: if True, use periodic displacement on torus in [-1, 1]
    
    Returns:
        c: [B, N] - neighbor counts
    """
    B, N, DIM = x.shape
    
    # Compute all pairwise distances
    r_vec = _pairwise_displacement(x, periodic, box_size)  # [B, N, N, DIM]
    
    # Count neighbors
    counts = torch.sum(count_kernel(r_vec, h), dim=2)  # [B, N]
    
    return counts


def blur(
    x: torch.Tensor,
    rho: torch.Tensor,
    A: torch.Tensor,
    h: float,
    mass: Optional[torch.Tensor] = None,
    periodic: bool = False,
    box_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Blur/smooth features using brute force pairwise comparison
    
    Args:
        x: [B, N, DIM] - particle positions
        rho: [B, N] - particle densities
        A: [B, N, F] - input features
        h: float - smoothing length
        mass: Optional[Tensor] of shape [B, N] or [B, N, 1]
        periodic: if True, use periodic displacement on torus in [-1, 1]
    
    Returns:
        SA: [B, N, F] - smoothed features
    """
    B, N, DIM = x.shape
    F = A.shape[2]
    
    # Convert density to volume (specific volume)
    m = _default_mass(x, mass)
    v = m / rho
    
    # Compute normalization
    normalization = smoothing_poly6_normalization(h, DIM)
    
    # Compute all pairwise distances
    r_vec = _pairwise_displacement(x, periodic, box_size)  # [B, N, N, DIM]
    
    # Compute smoothing kernel
    kernel_vals = smoothing_poly6_kernel(r_vec, h)  # [B, N, N]
    
    # Volume weighting
    v_j = v.unsqueeze(1)  # [B, 1, N]
    weighted_kernel = kernel_vals * v_j  # [B, N, N]
    
    # Smooth features
    A_j = A.unsqueeze(1)  # [B, 1, N, F]
    smoothed = torch.sum(weighted_kernel.unsqueeze(-1) * A_j, dim=2)  # [B, N, F]
    
    return normalization * smoothed


def density_gradient(
    x: torch.Tensor,
    h: float,
    mass: Optional[torch.Tensor] = None,
    periodic: bool = False,
    box_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute density gradients using brute force pairwise comparison
    
    Args:
        x: [B, N, DIM] - particle positions
        h: float - smoothing length
        mass: Optional[Tensor] of shape [B, N] or [B, N, 1]
        periodic: if True, use periodic displacement on torus in [-1, 1]
    
    Returns:
        Grho: [B, N, DIM] - density gradients
    """
    B, N, DIM = x.shape
    
    # Compute normalization
    normalization = gradient_spiky_normalization(h, DIM)
    
    m = _default_mass(x, mass)

    # Compute all pairwise distances
    r_vec = _pairwise_displacement(x, periodic, box_size)  # [B, N, N, DIM]
    
    # Compute gradient kernel
    grad_kernel = gradient_spiky_kernel(r_vec, h)  # [B, N, N, DIM]
    
    # Sum over neighbors (mass-weighted)
    m_j = m.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
    density_grads = torch.sum(grad_kernel * m_j, dim=2)  # [B, N, DIM]
    
    return normalization * density_grads

def moment_matrix(
    x: torch.Tensor,
    rho: torch.Tensor,
    h: float,
    mass: Optional[torch.Tensor] = None,
    periodic: bool = False,
    box_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute SPH moment matrix using brute force pairwise comparison
    
    The moment matrix M[i] = sum_j v_j * (x_j - x_i) ⊗ grad_W(x_j - x_i, h)
    where ⊗ denotes outer product and v_j is the volume of particle j.
    
    Args:
        x: [B, N, DIM] - particle positions
        rho: [B, N] - particle densities
        h: float - smoothing length
        mass: Optional[Tensor] of shape [B, N] or [B, N, 1]
        periodic: if True, use periodic displacement on torus in [-1, 1]
    
    Returns:
        M: [B, N, DIM, DIM] - moment matrices
    """
    B, N, DIM = x.shape
    
    # Convert density to volume (specific volume)
    m = _default_mass(x, mass)
    v = m / rho
    
    # Compute normalization
    normalization = gradient_spiky_normalization(h, DIM)
    
    # Compute all pairwise distances
    r_vec = _pairwise_displacement(x, periodic, box_size)  # [B, N, N, DIM] - (x_j - x_i)
    
    # Compute gradient kernel
    grad_kernel = gradient_spiky_kernel(r_vec, h)  # [B, N, N, DIM]
    
    # Volume weighting
    v_j = v.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
    weighted_grad = grad_kernel * v_j  # [B, N, N, DIM]
    
    # Compute outer products: (x_j - x_i) ⊗ (v_j * grad_W(x_j - x_i, h))
    # r_vec: [B, N, N, DIM], weighted_grad: [B, N, N, DIM]
    # outer product: [B, N, N, DIM, DIM]
    outer_products = r_vec.unsqueeze(-1) * weighted_grad.unsqueeze(-2)  # [B, N, N, DIM, DIM]
    
    # Sum over neighbors j to get moment matrix for each particle i
    moment_matrices = torch.sum(outer_products, dim=2)  # [B, N, DIM, DIM]
    
    return normalization * moment_matrices
