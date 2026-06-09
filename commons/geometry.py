import torch

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
    