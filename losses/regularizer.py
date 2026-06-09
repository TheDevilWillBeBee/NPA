import torch

def displacement_regularizer(ctx, return_info=False, return_summary=False):
    sum_dx = ctx.get('sum_dx')
    if sum_dx is not None:
        return sum_dx.mean()

    S = ctx.get('states')
    return torch.zeros((1,), dtype=S.dtype, device=S.device).mean()

def overflow_regularizer(ctx, return_info=False, return_summary=False):
    S = ctx.get('states')
    return (S - S.clip(-1.0, 1.0)).abs().mean()

def bound_regularizer(ctx, return_info=False, return_summary=False):
    x = ctx.get('positions')
    return (x - x.clip(-1.0, 1.0)).abs().mean()
