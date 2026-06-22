from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

import torchvision.transforms.functional as TF

from gsplat import rasterization as rasterize
from pytorch_msssim import ms_ssim, ssim # pip install pytorch-msssim


from .base import BaseLoss
from . import utils

def get_gaussian_properties_old(pos, state, **configs):
    s_iso = configs.get("sigma", 0.1)
    sh_degrees = configs.get("sh_degrees", None)
    max_anisotropy = configs.get("max_anisotropy", 3.0)
    opacity_scale = configs.get("opacity_scale", 1.0)

    means = pos[..., :3]

    colors_dim_start = 3 if sh_degrees is None else ((sh_degrees + 1) ** 2) * 3
    colors = state[..., -colors_dim_start:]
    if sh_degrees is not None:
        colors = colors.view(*colors.shape[:-1], -1, 3)
    else:
        colors = 0.5 + 0.5 * colors
        colors = colors + (colors.clamp(0.0, 1.0) - colors).detach()

    rotations_dim_start = colors_dim_start + 4
    rotations = state[..., -rotations_dim_start:-colors_dim_start]
    rotations = rotations + torch.randn_like(rotations) * 1e-6
    rotations = F.normalize(rotations, dim=-1)

    anisotropy_dim_start = rotations_dim_start + 3
    log_anisotropy = state[..., -anisotropy_dim_start:-rotations_dim_start]
    log_anisotropy = log_anisotropy + (log_anisotropy.clamp(-1.0, 1.0) - log_anisotropy).detach()
    log_anisotropy = max_anisotropy * log_anisotropy
    scales = s_iso * torch.exp(log_anisotropy)

    opacities = torch.exp(-log_anisotropy.sum(-1)) * opacity_scale

    return means, colors, opacities, scales, rotations, log_anisotropy

def get_gaussian_properties(pos, state, **configs):
    s_iso = configs.get("sigma", 0.1)
    sh_degrees = configs.get("sh_degrees", None)
    max_anisotropy = configs.get("max_anisotropy", 3.0)
    opacity_scale = configs.get("opacity_scale", 1.0)

    means = pos[..., :3]

    colors_dim_start = 3 if sh_degrees is None else ((sh_degrees + 1) ** 2) * 3
    colors = state[..., -colors_dim_start:]
    if sh_degrees is not None:
        colors = colors.view(*colors.shape[:-1], -1, 3)
    else:
        colors = 0.5 + 0.5 * colors
        colors = colors + (colors.clamp(0.0, 1.0) - colors).detach()

    if max_anisotropy == 0.0:
        # isotropic case
        rotations = torch.zeros_like(pos[..., :4])
        rotations[...,0] = 1.0  # w=1, x=y=z=0
        scales = s_iso * torch.ones_like(pos[..., :3])
        opacities = torch.ones_like(pos[..., 0]) * opacity_scale
        log_anisotropy = torch.zeros_like(pos[..., :4])
    else:
        rotations_dim_start = colors_dim_start + 4
        rotations = state[..., -rotations_dim_start:-colors_dim_start]
        rotations = rotations + torch.randn_like(rotations) * 1e-6
        rotations = F.normalize(rotations, dim=-1)

        anisotropy_dim_start = rotations_dim_start + 4
        log_anisotropy = state[..., -anisotropy_dim_start:-rotations_dim_start]
        # log_anisotropy = log_anisotropy + (log_anisotropy.clamp(-1.0, 1.0) - log_anisotropy).detach()
        log_anisotropy = F.tanh(log_anisotropy / max_anisotropy)
        log_anisotropy = max_anisotropy * log_anisotropy
        scales = s_iso * torch.exp(log_anisotropy[...,:3])

        opacities = torch.exp(log_anisotropy[...,3]) * opacity_scale

    return means, colors, opacities, scales, rotations, log_anisotropy

def rasterize_gaussians(
        pos, colors, opacities, scales, rotations,
        image_width, image_height, sh_degrees=None, 
        view_matrix=None, proj_matrix=None, 
        return_all=False, device=None
    ):
    viewmats = view_matrix
    Ks = proj_matrix

    render_results = rasterize(
        means=pos,
        quats=rotations,
        scales=scales,
        opacities=opacities,
        viewmats=viewmats,
        Ks=Ks,
        colors=colors,
        width=image_width,
        height=image_height,
        sh_degree=sh_degrees
    )
    if not return_all:
        render_results = (render_results[0], render_results[1])
    if device is not None:
        render_results = tuple(res.to(device) if isinstance(res, torch.Tensor) else res for res in render_results)
    return render_results

def photometric_loss(input_img, target_img, lambda_=0.2, levels=1, level_scale=4):
    target_img = target_img[None].expand_as(input_img) # Expand to match batch size

    input_img = input_img.reshape(-1, *input_img.shape[-3:]).permute(0,3,1,2)  # B, C, H, W
    target_img = target_img.reshape(-1, *target_img.shape[-3:]).permute(0,3,1,2)  # B, C, H, W

    if levels == 1:
        l1 = 0.0
        if lambda_ < 1.0:
            l1 = F.l1_loss(input_img, target_img)

        dssim = 0.0
        if lambda_ > 0:
            dssim = 1.0 - ms_ssim(input_img, target_img, data_range=1.0, size_average=True)
        return (1.0 - lambda_) * l1 + lambda_ * dssim

    total_loss = 0.0
    total_weight = 0.0
    current_input = input_img
    current_target = target_img
    for level in range(levels):
        level_weight = level_scale ** level
        l1 = 0.0
        if lambda_ < 1.0:
            l1 = F.l1_loss(current_input, current_target)
        dssim = 0.0
        if lambda_ > 0:
            dssim = 1.0 - ssim(current_input, current_target, data_range=1.0, size_average=True)
        level_loss = (1.0 - lambda_) * l1 + lambda_ * dssim

        total_loss += level_loss * level_weight
        total_weight += level_weight

        if level < levels - 1:
            current_input = TF.resize(
                current_input, 
                [current_input.shape[2] // level_scale, current_input.shape[3] // level_scale], 
                interpolation=TF.InterpolationMode.BICUBIC
            )
            current_target = TF.resize(
                current_target, 
                [current_target.shape[2] // level_scale, current_target.shape[3] // level_scale], 
                interpolation=TF.InterpolationMode.BICUBIC
            )
    return total_loss / level_weight

class GaussianSplat3DLoss(BaseLoss):
    def __init__(
        self,
        target,
        sigma=1.0,
        max_anisotropy=3.0,
        opacity_scale=1.0,
        sh_degrees=None,
        color_loss_weight=1.0,
        alpha_loss_weight=1.0,
        isotropy_regularization_weight=0.1,
        premultiplied_alpha=True,
        alpha_normalize=True,
        random_background=True,
        num_views=None,
        target_initial_sigma=0.0,
        target_final_sigma_t=0.0,
        regularizer_fn='mse_loss',
        photometric_loss_lambda=0.2,
        photometric_loss_levels=1,
        photometric_loss_level_scale=4,
    ):
        print(f'{sigma = }, {max_anisotropy = }, {opacity_scale = }')
        super().__init__()
        target_images = target["images"] # [C, H, W, 4]
        self.target_viewmats = target["viewmats"]  # [C, 4, 4]
        self.target_Ks = target["Ks"] # [C, 3, 3]
        self.target_colors = target_images[...,:3]
        self.target_alphas = target_images[...,3:4]

        if premultiplied_alpha:
            self.target_colors = self.target_colors * self.target_alphas

        self.color_loss_weight = color_loss_weight
        self.alpha_loss_weight = alpha_loss_weight
        self.isotropy_regularization_weight = isotropy_regularization_weight

        img_size = self.target_colors.shape[-3:-1]

        self.splat_kwargs = {
            "img_size": img_size,
            "sigma": sigma,
            "sh_degrees": sh_degrees,
            "max_anisotropy": max_anisotropy,
            "opacity_scale": opacity_scale,
        }

        self.alpha_normalize = alpha_normalize
        self.random_background = random_background

        self.total_views = target_images.shape[0]
        self.num_views = num_views

        self.target_initial_sigma = target_initial_sigma
        self.target_final_sigma_t = target_final_sigma_t

        self.regularizer_fn = {
            'mse_loss': F.mse_loss,
            'l1_loss': F.l1_loss,
        }[regularizer_fn]

        self.photometric_loss_params = {
            'lambda_': photometric_loss_lambda,
            'levels': photometric_loss_levels,
            'level_scale': photometric_loss_level_scale,
        }

    def splat_particles(self, pos, colors, opacities, scales, rotations, **configs):
        img_size = configs.get("img_size")
        width, height = img_size[1], img_size[0]
        sh_degrees = configs.get("sh_degrees", None)
        view_indices = configs.get("view_indices", slice(None))

        batch_size = pos.shape[0]
        view_matrix = self.target_viewmats
        view_matrix = view_matrix[None,view_indices].expand(batch_size, -1, -1, -1)
        
        proj_matrix = self.target_Ks
        proj_matrix = proj_matrix[None,view_indices].expand(batch_size, -1, -1, -1)

        render_results = rasterize_gaussians(
            pos, colors, opacities, scales, rotations,
            image_width=width,
            image_height=height,
            sh_degrees=sh_degrees,
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
        )

        render_color = render_results[0] # [B, C, H, W, 3]
        render_alpha = render_results[1] # [B, C, H, W, 1]

        return render_color, render_alpha

    def forward(self, ctx, return_info=True, return_summary=False):
        x, S = ctx["positions"], ctx["outputs"]
        t = ctx.get("progress", 1.0)
        gaussians = get_gaussian_properties(x, S, **self.splat_kwargs)

        pos, colors, opacities, scales, rotations, log_anisotropy = gaussians
        
        view_indices = slice(None)
        if self.num_views is not None:
            view_indices = torch.randperm(self.total_views)[:self.num_views]

        img_color, img_alpha = self.splat_particles(
            pos, colors, opacities, scales, rotations,
            **self.splat_kwargs,
            view_indices = view_indices
        )

        loss_target_color = self.target_colors[view_indices]
        loss_target_alpha = self.target_alphas[view_indices]

        if self.target_initial_sigma > 0.0 and t < self.target_final_sigma_t:
            current_sigma = self.target_initial_sigma * ((1.0 - (t / self.target_final_sigma_t)) ** 2)
            gaussian_kernel_size = int(2 * np.ceil(2 * current_sigma) + 1)
            gaussian_kernel_1d = torch.tensor(
                np.linspace(-gaussian_kernel_size // 2 + 1, gaussian_kernel_size // 2, gaussian_kernel_size),
                dtype=loss_target_color.dtype, device=loss_target_color.device,
            )
            gaussian_kernel_1d = torch.exp(-0.5 * (gaussian_kernel_1d / current_sigma) ** 2)
            gaussian_kernel_1d = gaussian_kernel_1d / gaussian_kernel_1d.sum()
            gaussian_kernel_2d = torch.outer(gaussian_kernel_1d, gaussian_kernel_1d).unsqueeze(0).unsqueeze(0)

            padding = gaussian_kernel_size // 2
            loss_target_color = F.conv2d(
                loss_target_color.permute(0,3,1,2), 
                gaussian_kernel_2d.expand(loss_target_color.shape[3], -1, -1, -1), 
                padding=padding, 
                groups=loss_target_color.shape[3]
            ).permute(0,2,3,1)
            loss_target_alpha = F.conv2d(
                loss_target_alpha.permute(0,3,1,2), 
                gaussian_kernel_2d.expand(loss_target_alpha.shape[3], -1, -1, -1), 
                padding=padding, 
                groups=loss_target_alpha.shape[3]
            ).permute(0,2,3,1)

        loss_img_color = img_color
        if self.random_background:
            bg_color = torch.rand_like(loss_target_color)
            loss_target_color = loss_target_color + (1.0 - loss_target_alpha) * bg_color
            loss_img_color = loss_img_color + (1.0 - img_alpha) * bg_color
        
        # color_weight = self.target_alpha.expand_as(img_color)
        # color_loss = F.mse_loss(color_loss_img_color, color_loss_target_color, weight=color_weight)
        color_loss = photometric_loss(loss_img_color, loss_target_color.detach(), **self.photometric_loss_params)

        alpha_scale = 1.0
        if self.alpha_normalize:
            alpha_scale = loss_target_alpha.mean(dim=(-3,-2), keepdim=True) / (img_alpha.mean(dim=(-3,-2), keepdim=True) + 1e-8)
            alpha_scale = alpha_scale.detach()
        alpha_loss = photometric_loss(img_alpha * alpha_scale, loss_target_alpha.detach(), **self.photometric_loss_params)
        isotropy_regularizer = self.regularizer_fn(log_anisotropy, torch.zeros_like(log_anisotropy))

        loss = color_loss * self.color_loss_weight
        loss += alpha_loss * self.alpha_loss_weight
        loss += isotropy_regularizer * self.isotropy_regularization_weight

        if return_info:
            with torch.no_grad():
                self.set_info(
                    loss.item(),
                    color_loss=(self.color_loss_weight, color_loss.item()),
                    alpha_loss=(self.alpha_loss_weight, alpha_loss.item()),
                    isotropy_regularizer=(
                        self.isotropy_regularization_weight,
                        isotropy_regularizer.item(),
                    ),
                )

        if return_summary:
            with torch.no_grad():
                
                c = 3
                cidx = torch.randperm(img_color.shape[1])[:c]
                b = min(2, img_color.shape[0])
                # bidx = torch.randperm(img_color.shape[0])[:b]
                bidx = torch.arange(b)

                target_img_color = loss_target_color[None,cidx].clamp(0.0, 1.0)
                target_img_color = target_img_color.repeat(b, 1, 1, 1, 1).reshape(b*c, *target_img_color.shape[-3:]).cpu().numpy()
                target_img_color = np.hstack(target_img_color * 255.0).astype(np.uint8)

                generated_img_color = loss_img_color[:b,cidx].clamp(0.0, 1.0)
                generated_img_color = generated_img_color.reshape(b*c, *generated_img_color.shape[-3:]).cpu().numpy()
                generated_img_color = np.hstack(generated_img_color * 255.0).astype(np.uint8)

                output_img_color = np.vstack([target_img_color, generated_img_color])

                max_alpha = max(loss_target_alpha.max().item(), img_alpha.max().item())
                target_img_alpha = loss_target_alpha[None,cidx].repeat(b, 1, 1, 1, 1).reshape(b*c, *loss_target_alpha.shape[-3:]).expand(-1,-1,-1,3).cpu().numpy()
                target_img_alpha = np.hstack((target_img_alpha / max_alpha) * 255.0).astype(np.uint8)

                generated_img_alpha = img_alpha[:b,cidx].reshape(b*c, *img_alpha.shape[-3:]).expand(-1,-1,-1,3).cpu().numpy()
                generated_img_alpha = np.hstack((generated_img_alpha / max_alpha).clip(0.0, 1.0) * 255.0).astype(np.uint8)

                output_img_density = np.vstack([target_img_alpha, generated_img_alpha])

                output_img = np.vstack([output_img_color, output_img_density])
                output_img = Image.fromarray(output_img)

                self.set_summary(image=output_img)

        return loss
