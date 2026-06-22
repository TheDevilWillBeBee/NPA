import os
import yaml
import argparse
import shutil

import numpy as np
import torch

from tqdm import trange
from datetime import datetime

from models.static_npa import NPA, TargetPool, step_euler
from sphops import HashGrid
from losses import Loss

import sphops

from commons import geometry
from utils import plotter

from logger import load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="configs/mnist.yaml", help="configuration"
)

from train import (
    get_optimizer,
    get_logger,
    fix_seed,
    flatten_config,
)

def get_dataset(device, cfg_train_dataset):
    import importlib
    import datasets
    
    module_name = cfg_train_dataset["name"]
    module_path = f"datasets.{module_name}"
    dataset_module = importlib.import_module(module_path)
    dataset = dataset_module.get_dataset(device=device, **cfg_train_dataset["kwargs"])
    return dataset

def sample_dataset(dataset, batch_size, idx=None, spatial_dims=2, state_dims=16, initial_state='zero'):
    if idx is None:
        batches = dataset[0].shape[0]
        idx = torch.randint(batches, (batch_size,))
    x, t = [arr[idx] for arr in dataset]

    spatial_dims_mem = 4 if spatial_dims == 3 else 2
    x = x[...,:spatial_dims_mem]
    s = torch.zeros(
        *x.shape[:-1], state_dims, dtype=x.dtype, device=x.device
    )
    if initial_state == 'position':
        s[..., :spatial_dims] = x[..., :spatial_dims]
    return x, s, t, idx

def main(cfg):
    flattened_config = flatten_config(cfg)

    device = torch.device(cfg["device"])
    print(f"Using device - {device}")

    if (rng_seed := cfg.get("rng_seed")) is not None:
        rng_seed = int(rng_seed)
        print(f"Using fixed seed - {rng_seed}")
        fix_seed(rng_seed)

    target = None # no global target
    train_data, test_data = get_dataset(device, cfg["train"]["dataset"])

    loss_fn = Loss(target, **cfg["train"]["loss"])

    logger = get_logger(cfg["train"].get("logger", {"backend": "disabled"}))

    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]
    step_range = cfg["train"]["step_range"]
    # inject_seed_interval = cfg["train"]["inject_seed_interval"]
    summary_interval = cfg["train"]["summary_interval"]
    num_particles = cfg["pool"]["num_particles"]
    
    spatial_dims = cfg["hashgrid"]["dim"]
    num_repetitions = cfg["train"]["num_repetitions"]
    # noise_std = cfg["train"]["noise_std"]

    use_pool = cfg["train"].get("use_pool", False)
    mutate_pool = cfg["train"].get("mutate_pool", False)
    stochastic_update = cfg["train"].get("stochastic_update", True)

    step_euler_kwargs = {}
    if not stochastic_update:
        step_euler_kwargs["update_prob"] = 1.0
        step_euler_kwargs["dt"] = 0.5

    validation_interval = cfg["train"]["validation"]["interval"]
    validation_model_log_interval = cfg["train"]["validation"].get("model_log_interval", None)
    validation_batch_size = cfg["train"]["validation"]["batch_size"]
    validation_loss_fn = Loss(target, **cfg["train"]["validation"]["loss"])
    validation_step_range = cfg["train"]["validation"].get("step_range", step_range)

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

    sampler_config = cfg["train"].get("sampler", {})
    def sample_dataset_fn(dataset, batch_size, idx=None):
        return sample_dataset(dataset, batch_size, idx=idx, spatial_dims=spatial_dims, state_dims=state_dims, **sampler_config)

    run_name = f"{config['experiment_name']}-{datetime.now():%m%d%H%M}"
    logger.start_run(run_name)
    logger.log_artifact(cfg["config_path"], "config.yaml")

    for k, v in flattened_config.items():
        logger.log_param(k, v)

    log_exit_status = "FAILED"
    try:
        for repetition in range(num_repetitions):
            pool = TargetPool(
                state_dims=state_dims,
                spatial_dims=spatial_dims,
                **cfg["pool"],
                device=device,
            )
            optimizer, scheduler = get_optimizer(model, cfg["train"])
            for epoch in (pbar := trange(epochs)):
                epoch = epoch + repetition * epochs
                model.train()
                # replace = (epoch % inject_seed_interval) == 0

                # x, s, t, idx = pool.sample(batch_size, replace_seed=replace)
                if use_pool:
                    x, s, t, idx = pool.sample(batch_size)

                    b_update = batch_size // 4
                    new_x, new_s, new_t, _ = sample_dataset_fn(train_data, b_update * 2)

                    x[:b_update] = new_x[:b_update]
                    s[:b_update] = new_s[:b_update]
                    t[:b_update] = new_t[:b_update]
                    
                    if mutate_pool:
                        x[-b_update:] = new_x[-b_update:]
                        # s[-b_update:] = new_s[-b_update:]
                        t[-b_update:] = new_t[-b_update:]
                    else:
                        x[-b_update:] = new_x[-b_update:]
                        s[-b_update:] = new_s[-b_update:]
                        t[-b_update:] = new_t[-b_update:]
                else:
                    x, s, t, idx = sample_dataset_fn(train_data, batch_size)

                steps = np.random.randint(*step_range)
                for _ in range(steps):
                    x, s = step_euler(model, x, s, grid, **step_euler_kwargs)
                out = model.decode(s)

                ctx = {
                    "positions": x,
                    "states": s,
                    "outputs": out,
                    "targets": t,
                }

                if use_pool:
                    with torch.no_grad():
                        pool.update(x, s, t, idx)

                return_info = (epoch % summary_interval) == 0 or logger.log_loss_every_step
                loss, loss_info = loss_fn(ctx, return_info=return_info)

                pbar.set_description(f"Repetition {repetition + 1}, Epoch {epoch + 1}")

                loss.backward()
                model.normalize_grads()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

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
                
                validate = (epoch % validation_interval) == 0
                if validate:
                    with torch.no_grad():
                        model.eval()
                        x_val, s_val, t_val, _ = sample_dataset_fn(test_data, validation_batch_size)
                        steps_val = np.random.randint(*validation_step_range)
                        for _ in range(steps_val):
                            x_val, s_val = step_euler(model, x_val, s_val, validation_grid)
                        out_val = model.decode(s_val)
                        ctx_val = {
                            "positions": x_val,
                            "states": s_val,
                            "outputs": out_val,
                            "targets": t_val,
                        }
                        val_loss, val_loss_info = validation_loss_fn(ctx_val, return_info=True)
                        val_loss_logs = {
                            f"Validation/{name}/{term_name}": term_value
                            for name, info in val_loss_info.items()
                            for term_name, (term_multiplier, term_value) in info.get("terms", {}).items()
                        }
                        logger.log_metrics(val_loss_logs, step=epoch)
                
                if validation_model_log_interval is not None and (epoch % validation_model_log_interval) == 0:
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

            # Full validation
            with torch.no_grad():
                model.eval()
                terms = []
                for idx in torch.split(torch.arange(test_data[0].shape[0]), validation_batch_size):
                    x_val, s_val, t_val, _ = sample_dataset_fn(test_data, idx.shape[0], idx=idx.to(device))
                    steps_val = step_range[1]
                    for _ in range(steps_val):
                        x_val, s_val = step_euler(model, x_val, s_val, validation_grid)
                    out_val = model.decode(s_val)
                    ctx_val = {
                        "positions": x_val,
                        "states": s_val,
                        "outputs": out_val,
                        "targets": t_val,
                    }
                    val_loss, val_loss_info = validation_loss_fn(ctx_val, return_info=True)
                    terms.append({
                        term_name: term_value
                        for name, info in val_loss_info.items()
                        for term_name, (term_multiplier, term_value) in info.get("terms", {}).items()
                    })
                mean_terms = {
                    k: np.mean([term[k] for term in terms])
                    for k in terms[0].keys()
                }
                logger.log_metrics({
                        f"Test/{k}": v for k, v in mean_terms.items()
                    }, step=(repetition + 1) * epochs - 1,
                )
        log_exit_status = "FINISHED"
    except KeyboardInterrupt:
        log_exit_status = "KILLED"
        print("Aborting training..")
    finally:
        logger.end_run(log_exit_status)

    


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = config["experiment_name"]
    exp_path = f"results/{exp_name}/"
    config["experiment_path"] = exp_path
    config["config_path"] = args.config
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    shutil.copy(f"{args.config}", f"{exp_path}/config.yaml")
    main(config)
