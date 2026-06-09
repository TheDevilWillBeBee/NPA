from PIL import Image
import numpy as np
import torch


from .base import BaseLoss
from . import utils

import sys, os

sys.path.insert(0, os.path.abspath(".."))

from utils.splatting import (
    splat_particles,
    splat_particles_batch,
)


class SplatLoss(BaseLoss):
    def __init__(
        self,
        target,
        grid_size=128,
        sigma=1.0,
        lo=-1.0,
        hi=1.0,
        normalize=True,
        color_loss_weight=100.0,
        density_loss_weight=1.0,
        normalize_density=False,
        lpips_loss_weight=0.0,
        center=True,
    ):
        super().__init__()
        target_pos = target["positions"]
        target_colors = target["colors"]
        pixel_size = target["pixel_size"]
        self.target_points = target_pos.shape[0]
        self.normalize = normalize
        self.lpips_loss_weight = lpips_loss_weight
        self.color_loss_weight = color_loss_weight
        self.density_loss_weight = density_loss_weight
        self.center = center
        self.normalize_density = normalize_density

        self.splat_kwargs = {
            "img_size": grid_size,
            "sigma": sigma,
            "lo": lo,
            "hi": hi,
            "normalize_color": True,
            "y_up": True,
            "pixel_size": pixel_size,
        }

        if self.lpips_loss_weight > 0:
            from lpips import LPIPS

            self.lpips_loss_fn = LPIPS(net="vgg").to(target_pos.device)

        with torch.no_grad():
            img_color, img_density = self.splat_particles(target_pos, target_colors)
            self.target_color = img_color
            self.target_density = img_density
            self.max_density = self.target_density.max().item()
            if self.normalize_density:
                self.target_density = self.target_density / self.max_density
            self.target_pos = target_pos
            self.target_colors = target_colors

    def splat_particles(self, x, colors):
        if len(x.shape) == 2:  # N 2
            img = splat_particles(x, colors, **self.splat_kwargs)
        elif len(x.shape) == 3:  # B N 2
            img = splat_particles_batch(x, colors, **self.splat_kwargs)

        if self.normalize:
            input_points = x.shape[-2]
            img = img * (self.target_points / input_points)

        if len(img.shape) == 3:
            img = img.unsqueeze(0)  # Add B=1
        img = img.permute(0, 3, 1, 2)  # B, C, H, W
        img_color, img_density = img[..., :3, :, :], img[..., 3:, :, :]

        return img_color, img_density

    def forward(self, ctx, return_info=True, return_summary=False):
        x, S = ctx["positions"], ctx["outputs"]

        if self.center:
            x = x = x - x.mean(dim=1, keepdim=True)
            x = x + self.target_pos.mean(dim=0)  # Shift to the center.

        particle_color = utils.get_color_from_state(S)
        img_color, img_density = self.splat_particles(x, particle_color)

        density_difference = utils.l1l2(img_density - self.target_density)
        color_loss = (
            utils.l1l2(img_color - self.target_color)
            * torch.exp(-1.0 * density_difference).detach()
        )
        color_loss = color_loss.mean()
        density_loss = density_difference.mean()

        loss = color_loss * self.color_loss_weight
        loss += density_loss * self.density_loss_weight

        if self.lpips_loss_weight > 0.0:
            lpips_loss = self.lpips_loss_fn(
                img_color, self.target_color, normalize=True
            ).mean()
            loss += lpips_loss * self.lpips_loss_weight

        if return_info:
            with torch.no_grad():
                self.set_info(
                    loss.item(),
                    color_loss=(self.color_loss_weight, color_loss.item()),
                    density_loss=(self.density_loss_weight, density_loss.item()),
                    lpips_loss=(
                        self.lpips_loss_weight,
                        lpips_loss.item() if self.lpips_loss_weight > 0 else 0.0,
                    ),
                )

        if return_summary:
            with torch.no_grad():
                b = min(4, img_color.shape[0])
                target_img_color = (
                    self.target_color.clamp(0.0, 1.0)
                    .repeat(b, 1, 1, 1)
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                )
                target_img_color = np.hstack(target_img_color * 255.0).astype(np.uint8)

                generated_img_color = (
                    img_color[:b].clamp(0.0, 1.0).cpu().numpy().transpose(0, 2, 3, 1)
                )
                generated_img_color = np.hstack(generated_img_color * 255.0).astype(
                    np.uint8
                )

                output_img_color = np.vstack([target_img_color, generated_img_color])

                max_density = max(
                    self.target_density.max().item(), img_density.max().item()
                )
                target_img_density = (
                    self.target_density.repeat(b, 3, 1, 1)
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                )
                target_img_density = np.hstack(
                    (target_img_density / max_density) * 255.0
                ).astype(np.uint8)

                generated_img_density = (
                    img_density[:b]
                    .repeat(1, 3, 1, 1)
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                )
                generated_img_density = np.hstack(
                    (generated_img_density / max_density).clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)

                output_img_density = np.vstack(
                    [target_img_density, generated_img_density]
                )

                output_img = np.vstack([output_img_color, output_img_density])
                output_img = Image.fromarray(output_img)

                self.set_summary(image=output_img)

        return loss
