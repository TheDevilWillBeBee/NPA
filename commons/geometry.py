import torch

from . import indexing

def grange(gshape, gmin, gsize, grid_offset=0.5, device=None):
    """
    gshape: i[N] = [gx, gy, ...]
    gmin: f[N]
    gsize: f[N]
    grid_offset: float or f[N]
    
    out: f[gx, gy, ..., N]
    """
    if device is None:
        device = gmin.device
    if not torch.is_tensor(gshape):
        gshape = torch.tensor(gshape, dtype=int, device=device)
    if not torch.is_tensor(gmin):
        gmin = torch.tensor(gmin, dtype=float, device=device)
    if not torch.is_tensor(gsize):
        gsize = torch.tensor(gsize, dtype=float, device=device)
    grid_a = [torch.arange(axis_shape, device=device) for axis_shape in gshape]
    grid_idx = torch.stack(torch.meshgrid(*grid_a, indexing='ij'), axis=-1).to(gmin.dtype)
    grid_pos = gmin + gsize * (grid_idx + grid_offset) / gshape
    return grid_pos

def _weighted_sample_grid2(gi, gp, grid, grid_shape, offset, dim, value_dims, grid_center_offset):
    """
    gi: i[*B, p, 2], lower left grid corner
    gp: f[*B, p, 2], sample position (relative to cell size)
    grid: f[*B, gx, gy, *n]
    grid_shape: i[2]
    offset: i[2]
    dim: int, grid.shape[dim] = gx
    value_dims: int, value_dims == |*n|
    grid_center_offset: float or f[2], offset from lower left corner to grid point, relative to cell size
    
    sample: f[*B, p, *n]
    """
    ogi = gi + offset # i[*B p 2], offset grid index 
    ogp = ogi + grid_center_offset # f[*B p 2], offset grid position (relative)
    w = torch.prod(1 - torch.abs(gp - ogp), axis=-1) # f[*B p], grid weight
    
    dim_start = dim
    dim_end = dim+2
    
    # clipped grid index
    cgi = torch.minimum(torch.maximum(ogi, torch.zeros_like(ogi, dtype=int)), grid_shape - 1).long()
    gv = indexing.batch_grid_indexing(grid, cgi, dim_start=dim_start, dim_end=dim_end) # f[*B p *n]
    
    return w[(...,) + ((None,) * value_dims)] * gv


def bilinear_sample(p, grid, gmin, gsize, dim=-2, grid_center_offset=0.5, cell_size=None):
    """
    p: f[*B, p, 2]
    grid: f[*B, gx, gy, *n]
    gmin: f[2]
    gsize: f[2]
    dim: int, grid.shape[dim] == gx
    grid_center_offset: float or f[2], offset from lower left corner to grid point, relative to cell size
    cell_size: optional, f[2]
    
    sample: f[*B, p, *n]
    """
    if dim < 0:
        dim = len(grid.shape) + dim
    
    batch_shape, grid_shape, value_size = grid.shape[:dim], grid.shape[dim:dim+2], grid.shape[dim+2:]
    batch_shape_p, num_points = p.shape[:-2], p.shape[-2]
    assert batch_shape == batch_shape_p
    
    grid_shape = torch.tensor(grid_shape, dtype=int, device=p.device)
    if cell_size is None:
        cell_size = gsize / grid_shape
    
    gp = (p - gmin) / cell_size # f[*B p 2], grid position [0-N]
    gi = torch.floor(gp - grid_center_offset).long() # i[*B p 2], lower left grid corner index
    
    value_dims = len(value_size)
    
    samples = [
        _weighted_sample_grid2(gi, gp, grid, grid_shape, torch.tensor([0,0], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid2(gi, gp, grid, grid_shape, torch.tensor([0,1], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid2(gi, gp, grid, grid_shape, torch.tensor([1,0], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid2(gi, gp, grid, grid_shape, torch.tensor([1,1], device=p.device), dim, value_dims, grid_center_offset),
    ]
    sample = torch.stack(samples, dim=-1).sum(dim=-1)
    
    return sample