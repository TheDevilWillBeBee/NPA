import numpy as np
import torch
import torch.nn.functional as F

from . import utils
from .base import BaseLoss

class PointwiseLoss(BaseLoss):
    def __init__(
        self,
        target, 
        output_dims: None | str | slice = None,
        loss_type: str = 'mse_loss',
        loss_fn_kwargs: dict = {}
    ):
        super().__init__()
        
        # target is ignored since we use the target from ctx

        if output_dims is None:
            output_dims = slice(None)
        elif isinstance(output_dims, str):
            output_dims = utils.parse_slice_string(output_dims)
        self.output_dims = output_dims

        assert loss_type in ['l1l2', 'l1_loss', 'mse_loss', 'cross_entropy'], f'Unsupported loss type: {loss_type}'
        self.loss_type = loss_type
        self.loss_fn_kwargs = loss_fn_kwargs

    def forward(self, ctx, return_info=True, return_summary=False):
        s = ctx["outputs"] # (B, N, C)
        t = ctx["targets"] # i(B), i(B, N), f(B, C2) or f(B, N, C2)

        output = s[..., self.output_dims]
        if self.loss_type in ['l1l2', 'l1_loss', 'mse_loss']:
            if t.dtype == torch.long:
                t = F.one_hot(t, num_classes=output.shape[-1]).to(output.dtype)
            if t.ndim == 2:
                t = t.unsqueeze(1).expand_as(output)
        elif self.loss_type in ['cross_entropy']:
            if t.dtype == torch.long:
                # t = [B], or [B, N]
                if t.ndim == 1:
                    t = t.unsqueeze(1).expand(-1, output.shape[1])
                # t = [B, N]
                t = t.reshape(-1)
            elif t.dtype == s.dtype:
                # t = [B, C2] or [B, N, C2]
                if t.ndim == 2:
                    t = t.unsqueeze(1).expand_as(output)
                # t = [B, N, C2]
                t = t.reshape(-1, t.shape[-1])
            output = output.view(-1, output.shape[-1])

        if self.loss_type == 'l1l2':
            loss = utils.l1l2(output - t).mean()
        else:
            loss_fn = getattr(F, self.loss_type)
            loss = loss_fn(
                output,
                t,
                **self.loss_fn_kwargs,
            )
        
        if return_info:
            term_info = {self.loss_type: (1.0, loss.item())}
            with torch.no_grad():
                self.set_info(
                    loss.item(),
                    **term_info
                )
        return loss
