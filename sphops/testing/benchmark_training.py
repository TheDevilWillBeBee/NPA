"""Matched graft-training benchmark for the growing and texture experiments.

This runs real optimizer updates, not a frozen-configuration microbenchmark.
Rendering, external loggers, and the separate stability rollout are excluded.
The normal training entry point and its defaults are left unchanged.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from losses import Loss
from models.npa import NPA, Pool, step_euler
from sphops import HashGrid
from sphops.testing.benchmark import GPUQuietGuard, fused_perception_context
from train import fix_seed, get_optimizer, get_target


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def create_state(cfg, checkpoint, device):
    fix_seed(cfg["rng_seed"])
    model = NPA(spatial_dims=cfg["hashgrid"]["dim"], **cfg["npa"]["kwargs"]).to(device)
    model.load_state_dict(checkpoint, strict=True)
    pool = Pool(state_dims=model.state_dims, spatial_dims=model.spatial_dims,
                device=device, **cfg["pool"])
    optimizer, scheduler = get_optimizer(model, cfg["train"])
    return model, pool, optimizer, scheduler


def run_epoch(epoch, cfg, model, pool, optimizer, scheduler, grid, loss_fn):
    """Same sampling/damage/rollout/loss/optimization/pool update as train.main."""
    settings = cfg["train"]
    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    torch.cuda.synchronize()
    started = time.perf_counter()
    events[0].record()
    with torch.no_grad():
        x, s, idx = pool.sample(settings["batch_size"], replace_seed=(
            epoch % settings["inject_seed_interval"] == 0))
        if settings["brush_size"] > 0:
            B, N, _ = x.shape
            random_point = torch.randint(N, (B,), device=x.device)
            center = x[torch.arange(B), random_point]
            distances = torch.norm(x - center.unsqueeze(1), dim=-1)
            s[distances < settings["brush_size"]] = 0.0
    steps = int(np.random.randint(*settings["step_range"]))
    events[1].record()
    sum_dx = 0.0
    for _ in range(steps):
        x, s, stats = step_euler(model, x, s, grid,
                               update_prob=settings["update_prob"], intermediate=True,
                               noise_std=settings["noise_std"])
        sum_dx += stats["dx_norm"]
    x = grid.wrap_positions(x)
    events[2].record()
    ctx = {"positions": x, "states": s, "outputs": model.decode(s), "sum_dx": sum_dx}
    loss, _ = loss_fn(ctx, return_info=False, return_summary=False)
    if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
    events[3].record()
    loss.backward()
    events[4].record()
    # Keep train.main's gradient-norm reads, even with its logger disabled.
    grad_norms = {name: float(param.grad.norm()) for name, param in model.named_parameters()
                  if param.grad is not None}
    gradients_finite = torch.stack([torch.isfinite(param.grad).all()
                                   for param in model.parameters()]).all()
    model.normalize_grads()
    optimizer.step()
    optimizer.zero_grad()
    if scheduler is not None:
        scheduler.step()
    with torch.no_grad():
        pool.update(x, s, idx)
    events[5].record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # Additional validation and recording happen outside the timed update.
    if not bool(gradients_finite):
        raise RuntimeError(f"Nonfinite raw gradients at epoch {epoch}")
    if not bool(torch.isfinite(x).all() & torch.isfinite(s).all()):
        raise RuntimeError(f"Nonfinite rollout at epoch {epoch}")
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError(f"Nonfinite model parameters at epoch {epoch}")
    phases = ("sampling", "rollout_forward", "loss_forward", "backward", "optimizer_pool")
    row = {"epoch": epoch + 1, "rollout_steps": steps, "loss": loss.item(),
           "wall_seconds": elapsed, "stream_ms": events[0].elapsed_time(events[-1]),
           "phase_stream_ms": {name: events[i].elapsed_time(events[i + 1])
                               for i, name in enumerate(phases)},
           "pool_indices": idx.cpu().tolist(), "grad_norms": grad_norms,
           "max_memory_allocated_bytes": torch.cuda.max_memory_allocated()}
    if epoch % 25 == 0 or epoch == settings["epochs"] - 1:
        with torch.no_grad():
            row["positions_rms_radius"] = float((
                (x - x.mean(dim=1, keepdim=True)).square().sum(-1).mean()).sqrt())
    return row


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "epochs.jsonl").exists() or (output / "model.pth").exists():
        raise RuntimeError(f"Refusing to overwrite an existing training run: {output}")
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["npa"]["graft_path"] = str(Path(args.graft).resolve())
    cfg["npa"]["kwargs"]["fused_perception"] = args.variant == "fused"
    if args.compile_step:
        cfg["npa"]["kwargs"]["compile_step"] = True
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["num_repetitions"] = 1
    cfg["train"]["target"]["path"] = str(Path(args.target).resolve())
    cfg["train"]["logger"] = {"backend": "disabled"}
    cfg["train"]["summary_interval"] = None
    cfg["train"]["stability_check_steps"] = 0
    if args.block_cap is not None:
        cfg["hashgrid"]["max_particles_per_block"] = args.block_cap
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    device = torch.device(cfg["device"])
    with GPUQuietGuard(device.index or 0) as guard:
        torch.cuda.set_device(device)
        from sphops.sph import sph_cuda
        extension_path = Path(sph_cuda.__file__).resolve()
        checkpoint = torch.load(args.graft, map_location=device, weights_only=True)
        target = get_target(device, cfg["train"]["target"])
        loss_fn = Loss(target, **cfg["train"]["loss"])
        grid = HashGrid(num_particles=cfg["pool"]["num_particles"],
                        batch_size=cfg["train"]["batch_size"], **cfg["hashgrid"])
        print(f"{args.variant}: {args.epochs} updates, graft={args.graft}, "
              f"target={args.target}, GPU={guard.telemetry()}", flush=True)
        model, pool, optimizer, scheduler = create_state(cfg, checkpoint, device)
        verified = []

        def verify_path(module, inputs):
            if verified or not inputs[0].requires_grad:
                return
            try:
                fused_perception_context(inputs[0])
                active = True
            except RuntimeError:
                active = False
            if active != (args.variant == "fused"):
                raise RuntimeError(f"Perception dispatch mismatch for {args.variant}")
            verified.append(active)

        # Validate native dispatch before compiling. A Python grad_fn traversal
        # inside a Linear hook cannot belong to the tensor-only compiled graph.
        if model.compile_step:
            sample_x = pool.pool_x[:cfg["train"]["batch_size"]]
            sample_s = pool.pool_s[:cfg["train"]["batch_size"]].detach().requires_grad_()
            probe, *_ = model.perceive(sample_x, sample_s, grid)
            verify_path(None, (probe,))
            del probe, sample_x, sample_s
            handle = None
        else:
            handle = model.model[0].register_forward_pre_hook(verify_path)
        warmup_started = time.perf_counter()
        for epoch in range(args.warmup_epochs):
            row = run_epoch(epoch, cfg, model, pool, optimizer, scheduler, grid, loss_fn)
            print(f"warmup {epoch + 1}: steps={row['rollout_steps']}, "
                  f"loss={row['loss']:.6f}, update={row['wall_seconds']:.3f}s", flush=True)
            guard.check()
        warmup_seconds = time.perf_counter() - warmup_started
        if handle is not None:
            handle.remove()
        if not verified:
            raise RuntimeError("Warmup did not validate the perception dispatch")
        # The callable captures this module's parameter identities. Warm the
        # same module used for timing, then restore it rather than recompiling
        # an independently constructed model inside the timed interval.
        compiled_model = model if model.compile_step else None
        del model, pool, optimizer, scheduler
        # TextureLoss has a VGG network outside the NPA optimizer. Discard its
        # warmup gradients too, without freezing it or changing train.main's
        # normal VGG backward/gradient-accumulation behavior in the real run.
        for loss_module in loss_fn.loss_modules.values():
            if isinstance(loss_module, torch.nn.Module):
                loss_module.zero_grad(set_to_none=True)
        gc.collect()
        # Start both real runs from the graft and an identically seeded fresh pool.
        model, pool, optimizer, scheduler = create_state(cfg, checkpoint, device)
        if compiled_model is not None:
            compiled_model.load_state_dict(model.state_dict(), strict=True)
            model = compiled_model
            optimizer, scheduler = get_optimizer(model, cfg["train"])
        metadata = {
            "variant": args.variant, "epochs": args.epochs, "config": cfg,
            "initial_model_sha256": tensor_hash(model.state_dict()),
            "initial_pool_sha256": tensor_hash({"x": pool.pool_x, "s": pool.pool_s}),
            "graft_file_sha256": hashlib.sha256(Path(args.graft).read_bytes()).hexdigest(),
            "target_file_sha256": hashlib.sha256(Path(args.target).read_bytes()).hexdigest(),
            "target_points": len(target["positions"]),
            "extension_path": str(extension_path),
            "extension_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
            "warmup_epochs": args.warmup_epochs, "warmup_seconds": warmup_seconds,
            "compile_step": model.compile_step,
            "dispatch_verified": True, "gpu_before": guard.telemetry(),
            "excluded": ["startup", "warmup", "rendering", "stability_evaluation", "checkpoint_io"],
            "phase_timing_note": "CUDA stream elapsed includes host launch gaps; not GPU-active time",
            "torch_version": torch.__version__,
            "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "loss_networks": {
                name: {"class": type(module).__name__,
                       "state_sha256": tensor_hash(module.state_dict()),
                       "parameter_count": sum(p.numel() for p in module.parameters()),
                       "requires_grad_parameter_count": sum(p.numel() for p in module.parameters()
                                                            if p.requires_grad)}
                for name, module in loss_fn.loss_modules.items()
                if isinstance(module, torch.nn.Module)
            },
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        guard.check()
        rows = []
        training_started = time.perf_counter()
        with (output / "epochs.jsonl").open("w") as log:
            for epoch in range(args.epochs):
                if guard.failure:
                    raise RuntimeError(guard.failure)
                row = run_epoch(epoch, cfg, model, pool, optimizer, scheduler, grid, loss_fn)
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if epoch == 0 or (epoch + 1) % 25 == 0:
                    guard.check()
                    elapsed = sum(r["wall_seconds"] for r in rows)
                    print(f"{args.variant} epoch {epoch + 1}/{args.epochs}: "
                          f"loss={row['loss']:.6f}, steps={row['rollout_steps']}, "
                          f"total_updates={elapsed:.2f}s, avg={elapsed / len(rows):.3f}s, "
                          f"GPU={guard.telemetry()}", flush=True)
        torch.cuda.synchronize()
        training_loop_seconds = time.perf_counter() - training_started
        guard.check()
        metadata.update({
            "completed_epochs": len(rows),
            "total_rollout_steps": sum(row["rollout_steps"] for row in rows),
            "update_wall_seconds": sum(row["wall_seconds"] for row in rows),
            "training_loop_wall_seconds": training_loop_seconds,
            "mean_update_wall_seconds": float(np.mean([row["wall_seconds"] for row in rows])),
            "median_update_wall_seconds": float(np.median([row["wall_seconds"] for row in rows])),
            "phase_stream_seconds": {
                key: sum(row["phase_stream_ms"][key] for row in rows) / 1000
                for key in rows[0]["phase_stream_ms"]},
            "final_loss": rows[-1]["loss"],
            "mean_last_25_loss": float(np.mean([row["loss"] for row in rows[-25:]])),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_after": guard.telemetry(), "foreign_gpu_process_detected": False,
            "final_model_sha256": tensor_hash(model.state_dict()),
            "all_loss_gradients_states_parameters_finite": True,
        })
        torch.save(model.state_dict(), output / "model.pth")
        (output / "summary.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({key: metadata[key] for key in (
            "variant", "completed_epochs", "total_rollout_steps", "update_wall_seconds",
            "training_loop_wall_seconds", "mean_update_wall_seconds", "final_loss",
            "mean_last_25_loss", "peak_memory_allocated_bytes")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/growing.yaml")
    parser.add_argument("--variant", choices=("fused", "unfused"), required=True)
    parser.add_argument("--graft", default="data/pretrained/lizard.pth")
    parser.add_argument("--target", required=True, help="Local emoji or texture PNG (no download)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--block-cap", type=int)
    parser.add_argument("--compile-step", action="store_true",
                        help="Opt-in NPA(compile_step=True): compile the MLP and Euler update")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.warmup_epochs < 1:
        parser.error("Epoch and warmup counts must be positive")
    run(args)
