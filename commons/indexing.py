import torch

def batch_grid_indexing(v, i, dim_start, dim_end=None):
    """
    v: tensor (*B, *m, *)
    i: int tensor (*B, *n, |*m|)
    out: tensor (*B, *n, *)
    
    out[j, *k, ...] = v[j, i[j, *k, 0], i[j, *k, 1], ... i[j, *k, |*m|], ...]
    """
    if dim_start < 0:
        dim_start = len(v.shape) + dim_start
    if dim_end is None:
        dim_end = len(v.shape)
    if dim_end < 0:
        dim_end = len(v.shape) + dim_end
    B, M, V = v.shape[:dim_start], v.shape[dim_start:dim_end], v.shape[dim_end:]
    Bi, N, lenM = i.shape[:dim_start], i.shape[dim_start:-1], i.shape[-1]

    assert B == Bi
    assert len(M) == lenM
    
    Bsize = 1
    for d in B:
        Bsize *= d
    
    Mcoeff = torch.ones(lenM, dtype=int, device=i.device)
    Msize = 1
    for d in range(lenM):
        Mcoeff[:d] *= M[d]
        Msize *= M[d]
    
    vv = v.view(-1, *V)
    if i.device.type == 'cuda':
        # workaround for RuntimeError: "addmm_cuda" not implemented for 'Long'
        ii = torch.inner(i.double(), Mcoeff.double()).long().view(Bsize, -1)
    else:
        ii = torch.inner(i, Mcoeff).view(Bsize, -1)
    ii = ii + torch.arange(Bsize, device=i.device)[:,None] * Msize
    
    return vv[ii].view(*B, *N, *V)
