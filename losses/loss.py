import inspect
import torch

from . import base
from . import utils

from .splat_loss import SplatLoss
from .texture_loss import TextureLoss

from .regularizer import *

def import_loss(name, cls=None):
    import importlib

    module_name = name
    class_name = utils.snake_to_pascal(name) if cls is None else cls

    try:
        module = importlib.import_module(f'.{module_name}', package=__package__)
        loss_cls = getattr(module, class_name, None)
        return loss_cls
    except ModuleNotFoundError:
        return None

class Loss(torch.nn.Module):
    def __init__(self, target, **kwargs):
        super().__init__()

        self.active_module_names = [k[:-7] for k in kwargs if k.endswith('_weight') and kwargs[k] != 0]
        self.module_weights = {
            name: kwargs.get(f'{name}_weight', 0.0)
            for name in self.active_module_names
        }
        self.module_configs = {
            name: kwargs.get(f'{name}_kwargs', None)
            for name in self.active_module_names
        }
        self.loss_modules = {
            name: self._get_loss_module(name, target, args)
            for name, args in self.module_configs.items()
        }

    def _get_loss_module(self, name, target, kwargs=None):
        loss_module = globals().get(name, None) 
        loss_module = globals().get(utils.snake_to_pascal(name), None) if loss_module is None else loss_module
        loss_module = import_loss(name) if loss_module is None else loss_module
        assert loss_module is not None, f'Loss module `{name}` not found!'
        
        if inspect.isfunction(loss_module):
            return loss_module
        if inspect.isclass(loss_module):
            if kwargs is None:
                kwargs = {}
            return loss_module(target, **kwargs)

    def _evaluate_loss(self, loss_module, ctx, return_info=True, return_summary=False):
        loss = loss_module(ctx, return_info=return_info, return_summary=return_summary)
        if not return_info and not return_summary:
            return loss, None
        
        if isinstance(loss_module, base.BaseLoss):
            return loss, loss_module.get_info()
        if inspect.isfunction(loss_module):
            return loss, {'value': loss.item(), 'terms': {}}

    def forward(self, input_dict, return_info=True, return_summary=False):
        loss = 0
        loss_info = {}
        for name, loss_module in self.loss_modules.items():
            if isinstance(loss_module, base.BaseLoss):
                loss_module.clear_info()
            l, info = self._evaluate_loss(loss_module, input_dict, return_info=return_info, return_summary=return_summary)
            if info is not None:
                loss_info[name] = info
            loss += l * self.module_weights[name]

        return loss, loss_info
