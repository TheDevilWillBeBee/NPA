import torch
import numpy as np
from sphops import (
    blur,
    gradient,
    moment_matrix,
    density_gradient,
    density,
    gradient_hybrid,
)
from sphops import bin_particles, unbin_particles, HashGrid


class NPA(torch.nn.Module):
    def __init__(
        self,
        spatial_dims=2, # We support particles in 2D and 3D
        state_dims=16, # Number of state channels for each particle. 
        hidden_dims=128, # Number of hidden neurons in the Adaptation MLP
        eps0=0.1, # Perception radius used for training the model
        alpha=1.0, # Scale factor for the output dx (position update)
        density_grad=True, # Perceive density gradient
        state_grad=True, # Perceive state gradient
        log_norm_grad=True, # Whether to log-normalize the state gradient
        log_norm_density_grad=True, # Whether to log-normalize the density gradient
        decoder_dims=None,  # Read-out MLP. Used in self-classifying digits experiment.
        output_dims=None, # Output dimensions for the read-out MLP. 
        stopgrad_pos=True, # Whether to Detach input position to the Perception
        stopgrad_state=False, # Whether to Detach input state to the Perception
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.state_dims = state_dims
        self.register_buffer("eps0", torch.tensor([eps0]))
        self.register_buffer("alpha", torch.tensor([alpha]))

        self.state_grad=state_grad
        self.density_grad = density_grad
        self.log_norm_grad = log_norm_grad
        self.log_norm_density_grad = log_norm_density_grad
        self.stopgrad_pos = stopgrad_pos
        self.stopgrad_state = stopgrad_state

        perception_dim = state_dims * 2
        if density_grad:
            perception_dim += spatial_dims
        if state_grad:
            perception_dim += spatial_dims * state_dims
        
        self.perception_dim = perception_dim

        self.decoder_dims = decoder_dims
        self.output_dims = output_dims

        self.model = self._get_model(spatial_dims, state_dims, hidden_dims)

        self.decoder = None
        if output_dims is not None:
            self.decoder = self._get_mlp(state_dims, output_dims, decoder_dims)

        self._initialize_weights()
    
    def _get_mlp(self, input_dims, output_dims, hidden_dims=None, last_bias=False):
        if hidden_dims is None:
            return torch.nn.Linear(input_dims, output_dims, bias=last_bias)
        return torch.nn.Sequential(
            torch.nn.Linear(input_dims, hidden_dims, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, output_dims, bias=last_bias),
        )
    
    def _get_model(self, spatial_dims, state_dims, hidden_dims):
        # Override here for variants
        output_dims = spatial_dims + state_dims
        return self._get_mlp(self.perception_dim, output_dims, hidden_dims)

    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.model[0].weight, gain=0.1)
        # torch.nn.init.zeros_(self.model[0].bias)
        torch.nn.init.xavier_uniform_(self.model[2].weight, gain=0.1)


    def _log_normalize_grad(self, grad_x):
        log_grad_norm = torch.log(1.0 + grad_x.norm(dim=-1, keepdim=True))
        return log_grad_norm * torch.nn.functional.normalize(grad_x, dim=-1)

    def perceive(self, x, s, grid: HashGrid):
        B, N, C = s.shape
        eps = grid.eps
        with torch.no_grad():
            snapshot = grid.instantiate(x)

        x_bin, s_bin = bin_particles(x, s, snapshot)

        if self.stopgrad_pos:
            x_perceive = x_bin.detach()
        else:
            x_perceive = x_bin

        if self.stopgrad_state:
            s_perceive = s_bin.detach()
        else:
            s_perceive = s_bin

        rho = density(x_perceive, snapshot)  # (B, N, 1)

        blur_s = blur(x_perceive, rho, s_perceive, snapshot)  # (B, N, C)
        
        z = [s_perceive, blur_s]
        
        if self.state_grad:
        
    #         grad_s = gradient(x_bin_detach, rho, s_bin, snapshot) # (B, N, C, dim)
            grad_s = gradient_hybrid(x_perceive, rho, s_perceive, snapshot)

            coef = eps / self.eps0

            grad_s = grad_s * coef

            if self.log_norm_grad:
                grad_s = self._log_normalize_grad(grad_s)

            z.append(grad_s.view(B, N, -1))

        if self.density_grad:
            grad_rho = density_gradient(x_perceive, snapshot)  # (B, N, dim)
            grad_rho = grad_rho * (coef ** (1 + self.spatial_dims))
            total_mass = N # Default mass of each particle is 1, so total mass is N. 
            grad_rho = grad_rho / total_mass # This is needed to make the grad_rho scale invariant to the number of particles.
            if self.log_norm_density_grad:
                grad_rho = self._log_normalize_grad(grad_rho)

            z.append(grad_rho)

        z = torch.cat(z, dim=-1)  # (B, N, C * (2 + dim) + dim)

        return z, x_bin, s_bin, snapshot

    def forward(self, x, s, grid: HashGrid):
        B, N, C = s.shape
        eps = grid.eps
        z, x_bin, s_bin, snapshot = self.perceive(x, s, grid)
        self.last_snapshot = snapshot

        y = self.model(z)  # (B, N, dim + C)
        dx_bin = y[..., : self.spatial_dims]
        dx_bin = (
            self.alpha * dx_bin * eps / (1.0 + dx_bin.norm(dim=-1, keepdim=True))
        )  # (B, N, dim)

        ds_bin = y[..., self.spatial_dims :]  # (B, N, C)

        return dx_bin, ds_bin, x_bin, s_bin

    def decode(self, s):
        if self.decoder is not None:
            return self.decoder(s)
        if self.output_dims is not None:
            return s[...,-self.output_dims:]
        return s

    def normalize_grads(self):
        for p in self.parameters():
            p.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            p.grad = p.grad / (p.grad.norm() + 1e-8) if p.grad is not None else p.grad


def step_euler(
    model,
    x,
    s,
    grid: HashGrid,
    dt=1.0,
    update_prob=0.5,
    intermediate=False,
    noise_std=0.0,
):
    # Update the particle state and position using the model's output. 
    # Note that the updated position and state do not have the same 
    # particle index as the input (the particles might be shuffled)
    B, N, C = s.shape
    dx_bin, ds_bin, x_bin, s_bin = model(x, s, grid)

    stats = None
    if intermediate:
        stats = {}
        stats["dx_norm"] = dx_bin.norm(dim=-1).mean()

    update_mask = 1.0
    if update_prob < 1.0:
        update_mask = (torch.rand(B, N, 1, device=x.device) < update_prob).float()

    if dx_bin.shape[-1] != x.shape[-1]:
        # For particles in 3D we actually pad them to 4D for better memory alignment
        dx_bin = torch.cat(
            [dx_bin, torch.zeros(B, N, 1, device=x.device)], dim=-1
        ).contiguous()

    x = x_bin + dt * dx_bin * update_mask
    s = s_bin + dt * ds_bin * update_mask
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std * np.sqrt(dt) * update_mask

    return x, s, stats


def step_euler_trace(
    model,
    x,
    s,
    grid: HashGrid,
    dt=1.0,
    update_prob=0.5,
    noise_std=0.0,
    return_aux=False,
):
    # Update the particle state and position using the model's output. 
    # The updated position and state with have the same particle index as the input
    B, N, C = s.shape
    dx_bin, ds_bin, x_bin, s_bin = model(x, s, grid)

    update_mask = 1.0
    if update_prob < 1.0:
        update_mask = (torch.rand(B, N, 1, device=x.device) < update_prob).float()

    if dx_bin.shape[-1] != x.shape[-1]:
        # For particles in 3D we actually pad them to 4D for better memory alignment
        dx_bin = torch.cat(
            [dx_bin, torch.zeros(B, N, 1, device=x.device)], dim=-1
        ).contiguous()

    x = x_bin + dt * dx_bin * update_mask
    s = s_bin + dt * ds_bin * update_mask
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std * np.sqrt(dt) * update_mask

    x_trace, s_trace = unbin_particles(x, s, model.last_snapshot)
    
    if return_aux:
        dx_trace, ds_trace = unbin_particles(dx_bin, ds_bin, model.last_snapshot)
        return x_trace, s_trace, dx_trace, ds_trace
    
    return x_trace, s_trace

class Pool:
    # Checkpoint pool used to store intermediate rollouts during training
    def __init__(
        self,
        device,
        pool_size=1,
        state_dims=8,
        num_particles=256,
        sigma=0.1,
        spatial_dims=2,
        seed_mode="uniform_circle",
        noise_level=0.0,
        random_seed=42,
    ):
        self.pool_size = pool_size
        self.num_particles = num_particles
        self.sigma = sigma
        self.noise_level = noise_level
        self.spatial_dims = spatial_dims
        self.state_dims = state_dims
        assert spatial_dims in [2, 3]
        self.seed_mode = seed_mode
        torch.manual_seed(random_seed)

        self.device = device

        self.seed_generator = self.get_seed_generator(seed_mode)
        self.pool_x, self.pool_s = self.seed_generator(pool_size)

    @torch.no_grad()
    def sample(self, batch_size, replace_seed=False):
        # Sample a batch of particles from the pool. 
        # If replace_seed, replace the first sample in the batch with a new random seed.
        batch_idx = torch.randperm(self.pool_size, device=self.device)[:batch_size]

        x = self.pool_x[batch_idx]
        s = self.pool_s[batch_idx]

        if replace_seed:
            new_x, new_s = self.seed_generator(1)
            x[:1, :, :] = new_x
            s[:1, :, :] = new_s

        return x, s, batch_idx

    @torch.no_grad()
    def update(self, x, s, batch_idx):
        # Return the updated particles back to the pool.
        self.pool_x[batch_idx, :, :] = x
        self.pool_s[batch_idx, :, :] = s

    def get_seed_generator(self, seed_mode):
        if seed_mode == "gaussian":
            return self.gaussian_seed
        elif seed_mode == "uniform":
            return self.uniform_seed
        elif seed_mode == "uniform_circle":
            return self.uniform_circle_seed
        else:
            raise ValueError(f"Unknown seed mode: {seed_mode}")

    def gaussian_seed(self, N):
        d = 2 if self.spatial_dims == 2 else 4
        x = torch.randn(N, self.num_particles, d, device=self.device) * self.sigma
        s = (
            torch.rand(N, self.num_particles, self.state_dims, device=self.device) - 0.5
        ) * self.noise_level
        return x, s

    def uniform_seed(self, N):
        d = 2 if self.spatial_dims == 2 else 4
        x = torch.rand(N, self.num_particles, d, device=self.device)
        x = (x - 0.5) * self.sigma * 2.0
        s = (
            torch.rand(N, self.num_particles, self.state_dims, device=self.device) - 0.5
        ) * self.noise_level
        return x, s

    def uniform_circle_seed(self, N):
        if self.spatial_dims == 2:
            d = 2
            theta = (
                torch.rand(N, self.num_particles, 1, device=self.device) * 2.0 * np.pi
            )
            r = (
                torch.sqrt(torch.rand(N, self.num_particles, 1, device=self.device))
                * self.sigma
            )
            x = torch.zeros(N, self.num_particles, d, device=self.device)
            x[..., 0] = r[..., 0] * torch.cos(theta[..., 0])
            x[..., 1] = r[..., 0] * torch.sin(theta[..., 0])
            s = (
                torch.rand(N, self.num_particles, self.state_dims, device=self.device)
                - 0.5
            ) * self.noise_level
            return x, s
        elif self.spatial_dims == 3:
            d = 4
            theta = (
                torch.rand(N, self.num_particles, 1, device=self.device) * 2.0 * np.pi
            )
            phi = torch.acos(
                2.0 * torch.rand(N, self.num_particles, 1, device=self.device) - 1.0
            )
            r = (
                torch.pow(
                    torch.rand(N, self.num_particles, 1, device=self.device), 1.0 / 3.0
                )
                * self.sigma
            )
            x = torch.zeros(N, self.num_particles, d, device=self.device)
            x[..., 0] = r[..., 0] * torch.sin(phi[..., 0]) * torch.cos(theta[..., 0])
            x[..., 1] = r[..., 0] * torch.sin(phi[..., 0]) * torch.sin(theta[..., 0])
            x[..., 2] = r[..., 0] * torch.cos(phi[..., 0])
            s = (
                torch.rand(N, self.num_particles, self.state_dims, device=self.device)
                - 0.5
            ) * self.noise_level
            return x, s
        else:
            raise ValueError(
                f"Unknown spatial dims for uniform_circle_seed: {self.spatial_dims}"
            )
