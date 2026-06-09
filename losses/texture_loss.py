import torch
import torchvision.models
import numpy as np
from PIL import Image

from . import utils
from .splat_loss import SplatLoss


class TextureLoss(SplatLoss):
    """https://arxiv.org/abs/1904.12785"""

    def __init__(
        self,
        target,
        grid_size=128,
        sigma=1.0,
        lo=-1.0,
        hi=1.0,
        normalize=True,
        n_samples=1024,
        color_loss_weight=100.0,
        density_loss_weight=1.0,
        style_layers=None,
        density_as_texture=False,
        normalize_density=False,
        replace_max_pool=True,
        density_power=1.0
    ):
        super().__init__(
            target=target,
            grid_size=grid_size,
            sigma=sigma,
            lo=lo,
            hi=hi,
            normalize=normalize,
            color_loss_weight=color_loss_weight,
            density_loss_weight=density_loss_weight,
            normalize_density=normalize_density,
        )

        self.style_layers = style_layers
        if self.style_layers is None:
            self.style_layers = [1, 6, 11, 18, 25]
        self.n_samples = n_samples
        self.density_power = density_power
        self.density_transformation = lambda x: (x + 1e-6) ** self.density_power
        with torch.no_grad():
            vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features.to(
                self.target_color.device
            )
            vgg_layers = []
            for l in vgg:
                if isinstance(l, torch.nn.MaxPool2d) and replace_max_pool:
                    l = torch.nn.AvgPool2d(
                        l.kernel_size, l.stride, l.padding, l.ceil_mode
                    )
                vgg_layers.append(l)
            self.vgg = torch.nn.Sequential(*vgg_layers)

            self.density_as_texture = density_as_texture
            if self.density_as_texture:
                # self.target_img = torch.cat(
                #     [self.target_color, self.target_density], dim=1
                # )  # [1,4,H,W]
                self.target_img = [
                    self.target_color,
                    self.density_transformation(self.target_density.repeat(1, 3, 1, 1)),
                ]

            else:
                self.target_img = [self.target_color]  # [1,3,H,W]

            self.target_features = [
                self.get_vgg_features(img) for img in self.target_img
            ]

    def get_vgg_features(self, imgs):
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)[:, None, None]
        x = (imgs - mean) / std
        b, c, h, w = x.shape
        features = []
        for i, layer in enumerate(self.vgg[: max(self.style_layers) + 1]):
            x = layer(x)
            if i in self.style_layers:
                b, c, h, w = x.shape
                features.append(x.reshape(b, c, h * w))
        return features

    @staticmethod
    def pairwise_distances_cos(x, y):
        x_norm = torch.norm(x, dim=2, keepdim=True)  # (b, n, 1)
        y_t = y.transpose(1, 2)  # (b, c, m) (m may be different from n)
        y_norm = torch.norm(y_t, dim=1, keepdim=True)  # (b, 1, m)
        dist = 1.0 - torch.matmul(x, y_t) / (x_norm * y_norm + 1e-10)  # (b, n, m)
        return dist

    @staticmethod
    def style_loss(x, y):
        pairwise_distance = TextureLoss.pairwise_distances_cos(x, y)
        m1, m1_inds = pairwise_distance.min(1)
        m2, m2_inds = pairwise_distance.min(2)
        remd = torch.max(m1.mean(dim=1), m2.mean(dim=1))
        return remd

    @staticmethod
    def moment_loss(x, y):
        mu_x, mu_y = torch.mean(x, 1, keepdim=True), torch.mean(y, 1, keepdim=True)
        mu_diff = torch.abs(mu_x - mu_y).mean(dim=(1, 2))

        x_c, y_c = x - mu_x, y - mu_y
        x_cov = torch.matmul(x_c.transpose(1, 2), x_c) / (x.shape[1] - 1)
        y_cov = torch.matmul(y_c.transpose(1, 2), y_c) / (y.shape[1] - 1)

        cov_diff = torch.abs(x_cov - y_cov).mean(dim=(1, 2))
        return mu_diff + cov_diff

    def ot_loss(self, generated_features, target_features):
        # Iterate over the VGG layers
        ot_loss = 0.0
        for x, y in zip(generated_features, target_features):
            (b_x, c, n_x), (b_y, _, n_y) = x.shape, y.shape
            n_samples = min(n_x, n_y, self.n_samples)
            indices_x = torch.argsort(torch.rand(b_x, 1, n_x, device=x.device), dim=-1)[
                ..., :n_samples
            ]
            x = x.gather(-1, indices_x.expand(b_x, c, n_samples))
            indices_y = torch.argsort(torch.rand(b_y, 1, n_y, device=y.device), dim=-1)[
                ..., :n_samples
            ]
            y = y.gather(-1, indices_y.expand(b_y, c, n_samples))
            x, y = x.transpose(1, 2), y.transpose(1, 2)  # (b, n_samples, c)
            ot_loss += self.style_loss(x, y) + self.moment_loss(x, y)

        return ot_loss

    def forward(self, ctx, return_info=True, return_summary=False):

        x, S = ctx["positions"], ctx["outputs"]
        batch_size, n_points = x.shape[0], x.shape[1]
        particle_color = utils.get_color_from_state(S)
        x = x - x.mean(dim=1, keepdim=True)
        img_color, img_density = self.splat_particles(x, particle_color)

        img_color = img_color * self.target_points / n_points
        img_density = img_density * self.target_points / n_points
        if self.normalize_density:
            img_density = img_density / self.max_density

        generated_img = [img_color]
        if self.density_as_texture:
            generated_img.append(self.density_transformation(img_density.repeat(1, 3, 1, 1)))

        generated_features = [self.get_vgg_features(img) for img in generated_img]
        if self.density_as_texture:
            density_loss = self.ot_loss(
                generated_features[1], self.target_features[1]
            ).mean()
        else:
            density_loss = utils.l1l2(img_density - self.target_density).mean()
        style_loss = self.ot_loss(generated_features[0], self.target_features[0]).mean()

        loss = (
            style_loss * self.color_loss_weight
            + density_loss * self.density_loss_weight
        )

        if return_info:
            with torch.no_grad():
                self.set_info(
                    loss.item(),
                    style_loss=(self.color_loss_weight, style_loss.item()),
                    density_loss=(self.density_loss_weight, density_loss.item()),
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

                max_density = max(self.max_density, img_density.max().item())
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
