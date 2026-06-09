import os
import yaml
import argparse
import shutil

import numpy as np
import torch

from tqdm import trange
from datetime import datetime

from models.npa import NPA, Pool, step_euler
from sphops import HashGrid
from losses import Loss

import sphops

from commons import geometry

from logger import load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="configs/growing.yaml", help="configuration"
)


@torch.no_grad()
def get_target(device, cfg_train_target):
    from utils import loader

    path = cfg_train_target["path"]
    threshold = cfg_train_target.get("threshold", 0.05)

    def get_image_size_from_num_points(num_points=4096):
        H = 128
        for _ in range(5):
            target_image = loader.load_target_image(path, size=[H, H])
            nonzero = target_image[..., 3] >= threshold
            nonzero = np.sum(nonzero)
            H = int(np.round(np.sqrt(num_points / nonzero) * H))

        return H

    image_size = cfg_train_target["image_size"]
    if image_size is None:
        H = get_image_size_from_num_points(cfg_train_target["num_points"])
        image_size = [H, H]
        print("Found adaptive image size of ", image_size)

    aabb = cfg_train_target["aabb"]

    pixel_size = (aabb[1] - aabb[0]) / image_size[0]

    aabb_min = aabb[0::2]
    aabb_max = aabb[1::2]

    target_image = loader.load_target_image(path, size=image_size)
    target_image = torch.from_numpy(
        target_image.transpose(1, 0, 2)[:, ::-1, :].copy()
    ).to(device)
    target_pos = geometry.grange(
        target_image.shape[:2],
        aabb_min,
        [max_ - min_ for min_, max_ in zip(aabb_min, aabb_max)],
        device=device,
    )

    target_alpha = target_image[..., 3]
    target_nonzero = target_alpha > threshold
    target_colors = target_image[target_nonzero][..., :3]
    target_pos = target_pos[target_nonzero]

    print("Number of target points: ", target_pos.shape)

    return {
        "image": target_image.to(torch.float32),
        "positions": target_pos.to(torch.float32),
        "colors": target_colors.to(torch.float32),
        "pixel_size": pixel_size,
    }


def get_optimizer(model, cfg_train):
    optimizer_cfg = cfg_train["optimizer"]
    optimizer_class = getattr(torch.optim, optimizer_cfg["name"])
    optimizer = optimizer_class(model.parameters(), **optimizer_cfg["kwargs"])

    scheduler_cfg = cfg_train.get("scheduler")
    scheduler = None
    if scheduler_cfg:
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_cfg["name"])
        scheduler = scheduler_class(optimizer, **scheduler_cfg["kwargs"])
    return optimizer, scheduler


def get_logger(cfg_logger):
    from logger import get_logger_class

    backend = cfg_logger["backend"]
    logger_class = get_logger_class(backend)
    return logger_class(**cfg_logger)


def fix_seed(seed=None):
    if seed is not None:
        import random

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


def flatten_config(cfg: dict, sep: str = "/"):
    leaves = {k: v for k, v in cfg.items() if not isinstance(v, dict)}
    nested_configs = {k: v for k, v in cfg.items() if isinstance(v, dict)}
    flattened_configs = [
        {k1 + sep + k2: v for k2, v in flatten_config(d, sep).items()}
        for k1, d in nested_configs.items()
    ]
    result = {**leaves}
    for c in flattened_configs:
        result.update(c)
    return result


def main(cfg):
    flattened_config = flatten_config(cfg)

    if os.path.exists(f'{cfg["experiment_path"]}/model.pth'):
        print("Experiment already exists! Aborting.")
        return

    device = torch.device(cfg["device"])
    print(f"Using device - {device}")

    if (rng_seed := cfg.get("rng_seed")) is not None:
        rng_seed = int(rng_seed)
        print(f"Using fixed seed - {rng_seed}")
        fix_seed(rng_seed)

    target = get_target(device, cfg["train"]["target"])

    loss_fn = Loss(target, **cfg["train"]["loss"])

    logger = get_logger(cfg["train"].get("logger", {"backend": "disabled"}))

    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]
    step_range = cfg["train"]["step_range"]
    inject_seed_interval = cfg["train"]["inject_seed_interval"]
    summary_interval = cfg["train"]["summary_interval"]
    num_particles = cfg["pool"]["num_particles"]

    spatial_dims = cfg["hashgrid"]["dim"]
    num_repetitions = cfg["train"]["num_repetitions"]
    noise_std = cfg["train"]["noise_std"]
    update_prob = cfg["train"].get("update_prob", 0.5)
    brush_size = cfg["train"].get("brush_size", 0.0)

    grid = HashGrid(
        num_particles=num_particles, batch_size=batch_size, **cfg["hashgrid"]
    )

    grid_test = HashGrid(num_particles=num_particles, batch_size=1, **cfg["hashgrid"])
    model = NPA(spatial_dims=spatial_dims, **cfg["npa"]["kwargs"]).to(device)
    state_dims = model.state_dims
    rep_start = 0
    if (graft_path := cfg["npa"].get("graft_path")) is not None:
        graft_path = cfg["npa"]["graft_path"]
        if os.path.exists(f'{cfg["experiment_path"]}/model_0.pth'):
            for i in range(100):
                if os.path.exists(f'{cfg["experiment_path"]}/model_{i}.pth'):
                    rep_start = i + 1
                    graft_path = f'{cfg["experiment_path"]}/model_{i}.pth'
                else:
                    break
            print(f"Resuming from repetition {rep_start}")

        checkpoint = load_checkpoint(graft_path, device)
        # Make sure eps0 and alpha are the same in the checkpoint and the current config, otherwise grafting will fail.
        if model.eps0.item() != checkpoint['eps0']:
            print("Warning: The pretrained model has a different eps0 value.")
        if model.alpha.item() != checkpoint['alpha']:
            print("Warning: The pretrained model has a different alpha value.")
        model.load_state_dict(checkpoint)
        print(f"Grafted model from {graft_path}")

    run_name = f"{cfg['experiment_name']}-{datetime.now():%m%d%H%M}"
    logger.start_run(run_name)
    logger.log_artifact(cfg["config_path"], "config.yaml")

    for k, v in flattened_config.items():
        logger.log_param(k, v)

    log_exit_status = "FAILED"
    try:
        for repetition in range(rep_start, num_repetitions):
            pool = Pool(
                state_dims=state_dims,
                spatial_dims=spatial_dims,
                **cfg["pool"],
                device=device,
            )
            optimizer, scheduler = get_optimizer(model, cfg["train"])
            for epoch in (pbar := trange(epochs + 1)):
                epoch = epoch + repetition * epochs
                replace = (epoch % inject_seed_interval) == 0

                with torch.no_grad():
                    x, s, idx = pool.sample(batch_size, replace_seed=replace)

                    # Erase circular regions of s based on the distance to a random point
                    if brush_size > 0.0:
                        B, N, D = x.shape
                        random_point = torch.randint(N, (B,), device=device)
                        center = x[torch.arange(B), random_point]  # [B, D]
                        distances = torch.norm(
                            x - center.unsqueeze(1), dim=-1
                        )  # [B, N]
                        mask = distances < brush_size
                        s = s.masked_fill(mask.unsqueeze(-1), 0.0)

                steps = np.random.randint(*step_range)
                sum_dx = 0.0
                for _ in range(steps):
                    x, s, stats = step_euler(
                        model,
                        x,
                        s,
                        grid,
                        update_prob=update_prob,
                        intermediate=True,
                        noise_std=noise_std,
                    )
                    sum_dx += stats["dx_norm"]
                x = grid.wrap_positions(x)

                out = model.decode(s)

                ctx = {
                    "positions": x,  # [B, N, D]
                    "states": s,  # [B, N, C]
                    "outputs": out,
                    "sum_dx": sum_dx,
                }

                summary = (epoch % summary_interval) == 0
                return_info = summary or logger.log_loss_every_step
                
                loss, loss_info = loss_fn(
                    ctx, return_info=return_info, return_summary=summary
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError("Loss is NaN or Inf. Stopping training.")

                pbar.set_description(f"Repetition {repetition + 1}, Epoch {epoch + 1}")

                loss.backward()
                with torch.no_grad():
                    grad_logs = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_logs[name] = param.grad.norm().item()
                    logger.log_metrics(
                        {f"Grad/{k}": v for k, v in grad_logs.items()}, step=epoch
                    )

                model.normalize_grads()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                with torch.no_grad():
                    if "textuer_loss" in loss_info:
                        x = x - x.mean(dim=1, keepdim=True)
                    pool.update(x, s, idx)

                    if return_info:
                        loss_logs = {
                            "Loss": loss.item(),
                            **{
                                f"Loss/{name}": info["value"]
                                for name, info in loss_info.items()
                            },
                            **{
                                f"Loss/{name}/terms/{term_name}": term_value
                                for name, info in loss_info.items()
                                for term_name, (
                                    term_multiplier,
                                    term_value,
                                ) in info.get("terms", {}).items()
                            },
                        }
                        logger.log_metrics(loss_logs, step=epoch)

                    if summary:

                        img = None
                        for key in ["splat_loss", "gaussian_splat_2d_loss", "texture_loss"]:
                            if key in loss_info:
                                img = loss_info[key]["summary"]["image"]
                                break
                        if img is not None:
                            logger.log_image(img, "train_sample", step=epoch)

                        if (
                            epoch % cfg["train"].get("stability_check_interval", 500)
                            == 0
                        ):
                            x, s, idx = pool.sample(1, replace_seed=True)
                            #                         for _ in tqdm(range(512)):
                            for test_timestep in trange(
                                cfg["train"].get("stability_check_steps", 4096)
                            ):
                                x, s, stats = step_euler(
                                    model,
                                    x,
                                    s,
                                    grid_test,
                                    update_prob=update_prob,
                                    intermediate=False,
                                    noise_std=0.0,
                                )

                                if test_timestep % 128 == 0:
                                    x = grid.wrap_positions(x)

                            if "splat_loss" in loss_info:
                                x = x - x.mean(dim=1, keepdim=True)
                                x = x + target["positions"].mean(
                                    dim=0
                                )  # Shift to the target center.

                            out = model.decode(s)
                            ctx = {
                                "positions": x,  # [B, N, D]
                                "states": s,  # [B, N, C]
                                "outputs": out,
                                "sum_dx": sum_dx * 0.0,
                            }
                            loss_test, loss_info_test = loss_fn(
                                ctx, return_info=True, return_summary=True
                            )

                            loss_logs_test = {
                                "Test Loss": loss_test.item(),
                                **{
                                    f"Test Loss/{name}": info["value"]
                                    for name, info in loss_info_test.items()
                                },
                            }
                            logger.log_metrics(loss_logs_test, step=epoch)

                            img = None
                            for key in ["splat_loss", "gaussian_splat_2d_loss", "texture_loss"]:
                                if key in loss_info:
                                    img = loss_info[key]["summary"]["image"]
                                    break

                            if img is not None:
                                logger.log_image(img, "test_sample", step=epoch)

            torch.save(
                model.state_dict(),
                f'{cfg["experiment_path"]}/model_{repetition}.pth',
            )
            logger.log_artifact(
                f'{cfg["experiment_path"]}/model_{repetition}.pth',
                f"model_{repetition}.pth",
            )
        log_exit_status = "FINISHED"
        torch.save(model.state_dict(), f'{cfg["experiment_path"]}/model.pth')
    except KeyboardInterrupt:
        log_exit_status = "KILLED"
        print("Aborting training..")
    finally:
        logger.end_run(log_exit_status)


def run(args, config):
    exp_name = config["experiment_name"]
    exp_path = f"results/{exp_name}/"
    config["experiment_path"] = exp_path
    config["config_path"] = args.config
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    shutil.copy(f"{args.config}", f"{exp_path}/config.yaml")
    main(config)


def emoji_to_identifier(ch: str) -> str:
    import unicodedata, re

    try:
        name = unicodedata.name(ch, "UNKNOWN")
    except:
        name = "UNKNOWN"
    # lower, replace spaces and hyphens with underscores
    ident = name.lower()
    ident = ident.replace(" ", "_").replace("-", "_")
    # remove anything that’s not alnum or underscore
    ident = re.sub(r"[^0-9a-z_]", "", ident)
    return ident


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = config["experiment_name"]
    target = config['train']["target"]["path"]
    target_identifier = emoji_to_identifier(target)
    if target_identifier != "unknown":
        exp_name += f"-{target_identifier}"
    else:
        target_identifier = os.path.splitext(os.path.basename(target))[0]
        exp_name += f"-{target_identifier}"
    config["experiment_name"] = exp_name
    run(args, config)

