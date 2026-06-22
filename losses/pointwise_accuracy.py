import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseLoss
from .utils import parse_slice_string

class PointwiseAccuracy(BaseLoss):
    def __init__(
        self,
        target, 
        output_dims: None | str | slice = None,
    ):
        super().__init__()
        
        # target is ignored since we use the target from ctx

        if output_dims is None:
            output_dims = slice(None)
        elif isinstance(output_dims, str):
            output_dims = parse_slice_string(output_dims)
        self.output_dims = output_dims

    def forward(self, ctx, return_info=True, return_summary=False):
        s = ctx["outputs"] # (B, N, C)
        t = ctx["targets"] # i(B), i(B, N), f(B, C2) or f(B, N, C2)

        output = s[..., self.output_dims]
        pred = output.argmax(dim=-1)
        if t.dtype != torch.long:
            t = t.argmax(dim=-1)
        is_global = (t.ndim == 1)
        global_t = t
        if t.ndim == 1:
            t = t.unsqueeze(1).expand_as(pred)
        # t = [B, N]
        
        correct = (pred == t).float()
        loss = (1.0 - correct).mean()
        
        global_info = {}
        if is_global:
            global_pred = pred.mode(dim=1).values
            correct_global = (global_pred == global_t).float()
            global_info['global_accuracy'] = (1.0, correct_global.mean().item())
            global_info['global_error'] = (1.0, (1.0 - correct_global).mean().item())
    
        if return_info:
            with torch.no_grad():
                self.set_info(
                    loss.item(),
                    accuracy=(1.0, correct.mean().item()),
                    error=(1.0, loss.item()),
                    **global_info,
                )
        return loss
