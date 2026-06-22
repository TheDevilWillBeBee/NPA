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

from logger import load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="configs/growing.yaml", help="configuration"
)


def get_target(device, cfg_train_target):
    from datasets import nerf_synthetic
    return nerf_synthetic.get_dataset(**cfg_train_target, device=device)

from train import (
    get_optimizer,
    get_logger,
    fix_seed,
    flatten_config,
)


def main(cfg):
    flattened_config = flatten_config(cfg)

    device = torch.device(cfg["device"])
    print(f"Using device - {device}")

    if (rng_seed := cfg.get("rng_seed")) is not None:
        rng_seed = int(rng_seed)
        print(f"Using fixed seed - {rng_seed}")
        fix_seed(rng_seed)

    target = get_target(device, cfg["train"]["target"])
    cfg_gaussian_splat = cfg["gaussian_splat"]

    cfg_loss = cfg["train"]["loss"]
    if "gaussian_splat_3d_loss_kwargs" in cfg_loss:
        cfg_loss["gaussian_splat_3d_loss_kwargs"].update(cfg_gaussian_splat)

    loss_fn = Loss(target, **cfg_loss)

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

    validation_target = get_target(device, cfg["train"]["validation"]["target"])
    validation_interval = cfg["train"]["validation"]["interval"]
    validation_batch_size = cfg["train"]["validation"]["batch_size"]
    validation_step_range = cfg["train"]["validation"].get("step_range", step_range)

    cfg_validation_loss = cfg["train"]["validation"]["loss"]
    if "gaussian_splat_3d_loss_kwargs" in cfg_validation_loss:
        cfg_validation_loss["gaussian_splat_3d_loss_kwargs"].update(cfg_gaussian_splat)
    validation_loss_fn = Loss(validation_target, **cfg_validation_loss)

    grid = HashGrid(
        num_particles=num_particles, batch_size=batch_size, **cfg["hashgrid"]
    )
    validation_grid = HashGrid(
        num_particles=num_particles, batch_size=validation_batch_size, **cfg["hashgrid"]
    )
    model = NPA(spatial_dims=spatial_dims, **cfg["npa"]['kwargs']).to(device)
    state_dims = model.state_dims
    if (graft_path := cfg["npa"].get("graft_path")) is not None:
        model.load_state_dict(load_checkpoint(graft_path, device))
        print(f'Grafted model from {graft_path}')

    run_name = f"{config['experiment_name']}-{datetime.now():%m%d%H%M}"
    logger.start_run(run_name)
    logger.log_artifact(cfg["config_path"], "config.yaml")

    for k, v in flattened_config.items():
        logger.log_param(k, v)

    log_exit_status = "FAILED"
    try:
        for repetition in range(num_repetitions):
            pool = Pool(
                state_dims=state_dims,
                spatial_dims=spatial_dims,
                **cfg["pool"],
                device=device,
            )
            optimizer, scheduler = get_optimizer(model, cfg["train"])
            for epoch in (pbar := trange(epochs)):
                epoch = epoch + repetition * epochs
                replace = (epoch % inject_seed_interval) == 0

                x, s, idx = pool.sample(batch_size, replace_seed=replace)

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
                        noise_std=noise_std
                    )
                    sum_dx += stats["dx_norm"]
                out = model.decode(s)

                ctx = {
                    "positions": x,
                    "states": s,
                    "outputs": out,
                    "sum_dx": sum_dx,
                    "progress": epoch / epochs,
                }

                return_info = (epoch % summary_interval) == 0 or logger.log_loss_every_step
                summary = (epoch % summary_interval) == 0
                loss, loss_info = loss_fn(ctx, return_info=return_info, return_summary=summary)

                pbar.set_description(f"Repetition {repetition + 1}, Epoch {epoch + 1}")

                loss.backward()
                model.normalize_grads()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                with torch.no_grad():
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
                            for term_name, (term_multiplier, term_value) in info.get("terms", {}).items()
                        }
                    }
                    logger.log_metrics(loss_logs, step=epoch)
                if summary:
                    img = None
                    if 'splat_loss' in loss_info:
                        img = loss_info['splat_loss']['summary']['image']
                    elif 'gaussian_splat_2d_loss' in loss_info:
                        img = loss_info['gaussian_splat_2d_loss']['summary']['image']
                    elif 'gaussian_splat_3d_loss' in loss_info:
                        img = loss_info['gaussian_splat_3d_loss']['summary']['image']
                    # img = plotter.plot_particles_splat_batch(x, s)
                    if img is not None:
                        logger.log_image(img, "step_sample", step=epoch)
                
                validate = (epoch % validation_interval) == 0
                if validate:
                    with torch.no_grad():
                        model.eval()
                        x_val, s_val, = pool.seed_generator(validation_batch_size)
                        steps_val = np.random.randint(*validation_step_range)
                        for _ in range(steps_val):
                            x_val, s_val, _ = step_euler(model, x_val, s_val, validation_grid)
                        out_val = model.decode(s_val)
                        ctx_val = {
                            "positions": x_val,
                            "states": s_val,
                            "outputs": out_val,
                        }
                        val_loss, val_loss_info = validation_loss_fn(ctx_val, return_info=True, return_summary=True)
                        val_loss_logs = {
                            f"Validation/{name}/{term_name}": term_value
                            for name, info in val_loss_info.items()
                            for term_name, (term_multiplier, term_value) in info.get("terms", {}).items()
                        }
                        logger.log_metrics(val_loss_logs, step=epoch)
                        logger.log_image(
                            val_loss_info['gaussian_splat_3d_loss']['summary']['image'], 
                            "validation_sample", 
                            step=epoch,
                        )
                        torch.save(
                            model.state_dict(),
                            f'{config["experiment_path"]}/model_{epoch}_{repetition}.pth',
                        )
                        logger.log_artifact(
                            f'{config["experiment_path"]}/model_{epoch}_{repetition}.pth',
                            f"checkpoints/model_{epoch}_{repetition}.pth",
                        )

            torch.save(
                model.state_dict(),
                f'{config["experiment_path"]}/model_{repetition}.pth',
            )
            logger.log_artifact(
                f'{config["experiment_path"]}/model_{repetition}.pth',
                f"model_{repetition}.pth",
            )
        log_exit_status = "FINISHED"
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



if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = config["experiment_name"]
    target = config['train']["target"]["rootdir"]
    target_identifier = os.path.splitext(os.path.basename(target))[0]
    exp_name += f"-{target_identifier}"
    config["experiment_name"] = exp_name
    run(args, config)

