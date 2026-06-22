import torch
import numpy as np
from sphops import blur, gradient, moment_matrix, density_gradient, density, gradient_hybrid
from sphops import bin_particles, HashGrid

from .npa import NPA as BaseNPA

class NPA(BaseNPA):
    def _get_model(self, spatial_dims, state_dims, hidden_dims):
        # Override here for variants
        input_dims = state_dims * (2 + spatial_dims) + spatial_dims
        output_dims = 0 + state_dims # Only update state dimensions
        return self._get_mlp(input_dims, output_dims, hidden_dims)
    
    def forward(self, x, s, grid: HashGrid):
        B, N, C = s.shape
        eps = grid.eps
        z, x_bin, s_bin, snapshot = self.perceive(x, s, grid)
        
        ds_bin = self.model(z) # (B, N, C)

        return ds_bin, x_bin, s_bin

def step_euler(model, x, s, grid: HashGrid, dt=1.0, update_prob=0.5, channel_mask=None):
    B, N, C = s.shape
    ds_bin, x_bin, s_bin = model(x, s, grid)

    update_mask = 1.0  
    if update_prob < 1.0:
        update_mask = (torch.rand(B, N, 1, device=x.device) < update_prob).float()

    if channel_mask is not None:
        ds_bin = ds_bin * channel_mask

    x = x_bin + 0
    s = s_bin + dt * ds_bin * update_mask

    return x, s

class TargetPool:
    def __init__(self, device, pool_size=1, num_particles=256, spatial_dims=2, state_dims=8, target_dtype=torch.long, target_shape=tuple()):
        self.pool_size = pool_size
        self.num_particles = num_particles
        self.spatial_dims = spatial_dims
        self.state_dims = state_dims
        assert spatial_dims in [2, 3]
        self.target_dtype = target_dtype
        self.target_shape = target_shape

        self.device = device

        self.spatial_dims_mem = 4 if spatial_dims == 3 else 2

        self.pool_x = torch.zeros(self.pool_size, self.num_particles, self.spatial_dims_mem, device=self.device)
        self.pool_s = torch.zeros(self.pool_size, self.num_particles, self.state_dims, device=self.device)
        self.pool_t = torch.zeros(self.pool_size, *target_shape, dtype=self.target_dtype, device=self.device)
        
    @torch.no_grad()
    def sample(self, batch_size):
        batch_idx = torch.randperm(self.pool_size, device=self.device)[:batch_size]

        x = self.pool_x[batch_idx]
        s = self.pool_s[batch_idx]
        t = self.pool_t[batch_idx]

        return x, s, t, batch_idx
    
    @torch.no_grad()
    def update(self, x, s, t, batch_idx):
        if x is not None:
            self.pool_x[batch_idx] = x
        if s is not None:
            self.pool_s[batch_idx] = s
        if t is not None:
            self.pool_t[batch_idx] = t
