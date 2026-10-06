"""Compare SPH perception and NPA forward/backward on a fixed particle corpus.

--prepare-only is CPU-only. GPU runs refuse competing compute processes, use
interleaved variant order, and report wrapper latency separately from NCU metrics.
"""

import argparse
from collections import defaultdict
import gc
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading

import numpy as np

if __package__:
    from .benchmark_data import build_cases, neighbor_statistics
else:
    from benchmark_data import build_cases, neighbor_statistics


class GPUQuietGuard:
    """Advisory lock plus continuous foreign-context detection during measurements."""
    def __init__(self, device):
        self.device = device
        self.stop = threading.Event()
        self.failure = None

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        result = subprocess.run(
            ["nvidia-smi", "-i", str(self.device), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        foreign = [int(line.strip()) for line in result.stdout.splitlines()
                   if line.strip().isdigit() and int(line.strip()) != os.getpid()]
        if foreign:
            raise RuntimeError(f"GPU {self.device} has other compute processes {foreign}; measurement rejected")

    def telemetry(self):
        result = subprocess.run(
            ["nvidia-smi", "-i", str(self.device),
             "--query-gpu=uuid,name,driver_version,clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()

    def _monitor(self):
        while not self.stop.wait(0.25):
            try:
                self.check()
            except Exception as error:
                self.failure = str(error)
                return

    def __enter__(self):
        self.lock = open(f"/tmp/npa-sph-gpu-{self.device}.lock", "a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.check()
        except Exception:
            self.lock.close()
            raise
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.lock.close()


def event_latency(torch, function, iterations, warmup, guard):
    guard.check()
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    guard.check()
    before = guard.telemetry()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    milliseconds = start.elapsed_time(end) / iterations
    guard.check()
    return milliseconds, {"before": before, "after": guard.telemetry()}


def summarize(samples):
    values = np.asarray(samples)
    return {"median_ms": float(np.median(values)), "mean_ms": float(values.mean()),
            "p25_ms": float(np.percentile(values, 25)), "p75_ms": float(np.percentile(values, 75)),
            "samples_ms": samples}


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["block_cap"], row["operation"], row["variant"], row["pass"])].append(row)
    return [{
        "family": key[0], "block_cap": key[1], "operation": key[2], "variant": key[3], "pass": key[4],
        "targets": [row["case"] for row in group],
        "mean_target_median_ms": float(np.mean([row["median_ms"] for row in group])),
    } for key, group in grouped.items()]


def compare_variants(rows):
    paired = defaultdict(dict)
    for row in rows:
        paired[(row["case"], row["family"], row["block_cap"], row["operation"], row["pass"])][row["variant"]] = row["median_ms"]
    return [{"case": key[0], "family": key[1], "block_cap": key[2],
             "operation": key[3], "pass": key[4],
             "speedup": pair["legacy"] / pair["fused"]}
            for key, pair in paired.items() if "legacy" in pair and "fused" in pair]


def canonical_particles(values, permutation):
    """Binned tensors to original particle order, including autograd."""
    index = permutation.long().unsqueeze(-1)
    return tuple(value.gather(1, index.expand_as(value)) for value in values)


def binned_particles(values, permutation):
    """Original-order tensors/cotangents to a snapshot's bin permutation."""
    index = permutation.long().unsqueeze(-1)
    return tuple(value.new_empty(value.shape).scatter_(1, index.expand_as(value), value)
                 for value in values)


def numerical_difference(actual, expected):
    difference = (actual.detach() - expected.detach())
    return {"max_absolute_error": float(difference.abs().max()),
            "relative_l2_error": float(difference.norm() / expected.detach().norm().clamp_min(1e-12))}


def fused_perception_context(output):
    """Find the native context underneath differentiable padding/unpacking."""
    pending, visited = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        if type(node).__name__ == "FusedPerceptionBackward":
            return node
        pending.extend(parent for parent, _ in node.next_functions)
    raise RuntimeError("No fused perception context found; cannot validate its cached adjoint")


def cached_state_adjoint_reference(torch, x, state, snapshot, eps0, output, cotangent, mass=None):
    """Independent legacy SPH/PyTorch adjoint at the fused normalization point.

    log(1 + norm) rounds its addition before the log. A few ulps in a small
    SPH gradient can cross that boundary and change the normalization Jacobian.
    Hold the forward values fixed, not the Jacobian: legacy SPH backward and
    PyTorch autograd still compute the complete state derivative independently.
    """
    from sphops import blur, gradient
    from sphops.sph import _log_normalize_gradient
    context = fused_perception_context(output)
    _, volume, corrected_values, _, _ = context.saved_tensors
    from sphops.sph import _default_mass
    mass = _default_mass(x, mass).contiguous()
    rho = (mass / volume).detach()
    raw = gradient(x, rho, state, snapshot, mass=mass) * (snapshot.grid.eps / eps0)
    # Evaluate at exactly the cached values, with the original straight-through
    # identity derivative to raw SPH (including the radius scale).
    if context.log_norm_grad:
        corrected_values = corrected_values[..., :state.shape[-1], :]
        aligned = (raw - raw.detach()) + corrected_values.detach()
        gradient = _log_normalize_gradient(aligned)
    else:
        gradient = raw
    channels = state.shape[-1]
    parts = [state, blur(x, rho, state, snapshot, mass=mass), gradient.flatten(-2)]
    parts.append(output[..., channels * (2 + snapshot.dim):].detach())
    reference = torch.cat(parts, dim=-1)
    return torch.autograd.grad(reference, state, cotangent)[0]


def make_workloads(torch, case, cap, channels, hidden_dims, operations, variants, device,
                   mass_range=(1.0, 1.0)):
    # Import only for an explicitly requested GPU run. Tensor setup is untimed.
    from sphops import HashGrid, bin_particles, density, blur, gradient
    from sphops import density_gradient, moment_matrix, perception, fused_perception_supported
    from models.npa import NPA

    B, N, _ = case.positions.shape
    x = torch.from_numpy(case.positions).to(device)
    generator = torch.Generator(device=device).manual_seed(case.seed + 3000)
    state = (torch.zeros(B, N, channels, device=device) if case.family.endswith("seed")
             else torch.randn(B, N, channels, generator=generator, device=device) * 0.1)
    mass_generator = torch.Generator(device=device).manual_seed(case.seed + 4000)
    mass = (torch.full((B, N), mass_range[0], device=device) if mass_range[0] == mass_range[1]
            else torch.rand(B, N, generator=mass_generator, device=device)
            * (mass_range[1] - mass_range[0]) + mass_range[0])
    mass_info = {"range": list(mass_range), "seed": case.seed + 4000,
                 "min": float(mass.min()), "max": float(mass.max()), "mean": float(mass.mean()),
                 "sha256_original_order": hashlib.sha256(mass.cpu().numpy().tobytes()).hexdigest()}
    grid = HashGrid(dim=2, boundary=case.boundary, mode="grid", grid_size=case.grid_size,
                    eps=case.eps, num_particles=N, batch_size=B, max_particles_per_block=cap)
    snapshot = grid.instantiate(x)
    x_bin, s_bin, mass_bin = bin_particles(x, state, snapshot, mass=mass)
    eps0 = torch.tensor([case.eps], device=device)
    if "fused" in variants and not fused_perception_supported(x_bin, s_bin, snapshot, mass_bin):
        raise RuntimeError("Fused kernels unavailable; rebuild the CUDA extension or set NPA_SPH_EXTENSION_PATH to a new build")
    workloads = {}
    validation = {}
    if "perception" in operations:
        reference_output, reference_gradient = None, None
        for variant in variants:
            s = s_bin.detach().requires_grad_(True)
            fused = variant == "fused"
            def forward(s=s, fused=fused):
                return perception(x_bin, s, snapshot, eps0, mass=mass_bin, use_fused=fused)
            output = forward()
            cotangent = torch.randn(output.shape, generator=generator, device=device)
            # Reuse exactly the same incoming gradient across variants.
            if variant == variants[0]:
                perception_cotangent = cotangent
            def backward(output=output, s=s, cotangent=perception_cotangent):
                return torch.autograd.grad(output, s, cotangent, retain_graph=True)
            if fused:
                cached_reference_gradient = cached_state_adjoint_reference(
                    torch, x_bin, s, snapshot, eps0, output, perception_cotangent, mass=mass_bin)
                fused_gradient = backward()[0]
                torch.testing.assert_close(fused_gradient, cached_reference_gradient, rtol=5e-4, atol=2e-4)
                validation[("perception", variant)] = {
                    "logical_channels": channels,
                    "native_channels": fused_perception_context(output).state_shape[-1],
                    "padding_and_unpacking_included": True,
                    "cached_state_adjoint": numerical_difference(fused_gradient, cached_reference_gradient),
                    "cached_state_adjoint_tolerance": {"rtol": 5e-4, "atol": 2e-4},
                }
            if reference_output is None:
                reference_output = output.detach()
                reference_gradient = backward()[0]
            else:
                torch.testing.assert_close(output.detach(), reference_output, rtol=5e-4, atol=5e-5)
                actual_gradient = backward()[0]
                # Pointwise derivative correctness is checked independently at
                # identical cached values above. Also bound overall agreement
                # with the unmodified legacy normalization evaluation point.
                gradient_difference = numerical_difference(actual_gradient, reference_gradient)
                if gradient_difference["relative_l2_error"] > 1e-5:
                    raise AssertionError(f"SPH state-adjoint relative L2 error exceeds 1e-5: {gradient_difference}")
                pointwise_mismatches = int(((actual_gradient - reference_gradient).abs()
                                            > 2e-4 + 5e-4 * reference_gradient.abs()).sum())
                validation.setdefault(("perception", variant), {}).update({
                    "reference_variant": variants[0],
                    "output": numerical_difference(output, reference_output),
                    "state_gradient": gradient_difference,
                    "state_gradient_relative_l2_limit": 1e-5,
                    "original_normalization_pointwise_mismatches": pointwise_mismatches,
                })
                if pointwise_mismatches:
                    print(f"Numerical check {case.name}/cap={cap}: {pointwise_mismatches} original-point adjoint "
                          "differences; cached-point adjoint passed independently")
            workloads[("perception", variant)] = (forward, backward)
    if "npa" in operations:
        weights = None
        reference_output, reference_gradients, reference_preactivation = None, None, None
        for variant in variants:
            model = NPA(state_dims=channels, hidden_dims=hidden_dims, eps0=case.eps,
                        alpha=0.5, fused_perception=variant == "fused").to(device)
            if weights is None:
                weights = model.state_dict()
            else:
                model.load_state_dict(weights)
            s = state.detach().requires_grad_(True)
            def forward(model=model, s=s):
                return model(x, s, grid, mass=mass)[:2]
            captured = []
            hook = model.model[0].register_forward_hook(
                lambda module, inputs, output: captured.append(output.detach()))
            output = forward()
            hook.remove()
            canonical_output = canonical_particles(output, model.last_snapshot.permutation)
            preactivation, = canonical_particles(captured, model.last_snapshot.permutation)
            if variant == variants[0]:
                npa_cotangents = tuple(torch.randn(o.shape, generator=generator, device=device) for o in canonical_output)
            cotangents = binned_particles(npa_cotangents, model.last_snapshot.permutation)
            inputs = (s, *model.parameters())
            def backward(output=output, inputs=inputs, cotangents=cotangents):
                return torch.autograd.grad(output, inputs, cotangents, retain_graph=True)
            if reference_output is None:
                reference_output = tuple(value.detach() for value in canonical_output)
                reference_gradients = backward()
                reference_preactivation = preactivation
            else:
                for actual, expected in zip(canonical_output, reference_output):
                    torch.testing.assert_close(actual.detach(), expected, rtol=5e-4, atol=5e-5)
                switches = (preactivation > 0) != (reference_preactivation > 0)
                switch_count = int(switches.sum())
                switch_magnitude = (float(torch.maximum(preactivation.abs(), reference_preactivation.abs())[switches].max())
                                    if switch_count else 0.0)
                if switch_magnitude > 1e-5:
                    raise AssertionError(f"ReLU sign changes outside the roundoff interval: {switch_magnitude}")
                actual_gradients = backward()
                gradient_checks = []
                for actual, expected in zip(actual_gradients, reference_gradients):
                    metrics = numerical_difference(actual, expected)
                    check = "pointwise"
                    try:
                        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
                    except AssertionError:
                        # ReLU is discontinuous at zero: roundoff-sized changes
                        # in its input can change an entire neuron's derivative.
                        # SPH output/state-adjoint checks above stay pointwise.
                        if not switch_count:
                            raise
                        torch.testing.assert_close(
                            (actual - expected).norm(), expected.norm().new_zeros(()),
                            rtol=0, atol=max(1e-4, float(expected.norm()) * 1e-3),
                        )
                        check = "normwise_near_zero_relu"
                    gradient_checks.append({**metrics, "check": check})
                validation[("npa", variant)] = {
                    "reference_variant": variants[0],
                    "outputs": [numerical_difference(a, b) for a, b in zip(canonical_output, reference_output)],
                    "gradients": gradient_checks, "relu_mask_switches": switch_count,
                    "max_preactivation_at_switch": switch_magnitude,
                }
                if switch_count:
                    print(f"Numerical check {case.name}/cap={cap}: {switch_count} near-zero ReLU mask changes "
                          f"(max |input|={switch_magnitude:.2e})")
            workloads[("npa", variant)] = (forward, backward)

    # Standalone operations retain their original full backward, including rho
    # as an independent input. Perception/NPA above use actual stopgrad_pos=True.
    primitive_names = set(operations) - {"perception", "npa"}
    if primitive_names:
        rho = density(x_bin, snapshot, mass=mass_bin).detach()
        for name in sorted(primitive_names):
            position = x_bin.detach().requires_grad_(True)
            s = s_bin.detach().requires_grad_(True)
            r = rho.detach().requires_grad_(True)
            if name == "density":
                def forward(position=position):
                    return density(position, snapshot, mass=mass_bin)
                inputs = (position,)
            elif name == "density_gradient":
                def forward(position=position):
                    return density_gradient(position, snapshot, mass=mass_bin)
                inputs = (position,)
            elif name == "blur":
                def forward(position=position, r=r, s=s):
                    return blur(position, r, s, snapshot, mass=mass_bin)
                inputs = (position, r, s)
            elif name == "gradient":
                def forward(position=position, r=r, s=s):
                    return gradient(position, r, s, snapshot, mass=mass_bin)
                inputs = (position, r, s)
            elif name == "moment_matrix":
                def forward(position=position, r=r):
                    return moment_matrix(position, r, snapshot, mass=mass_bin)
                inputs = (position, r)
            output = forward()
            cotangent = torch.randn(output.shape, generator=generator, device=device)
            def backward(output=output, inputs=inputs, cotangent=cotangent):
                return torch.autograd.grad(output, inputs, cotangent, retain_graph=True)
            workloads[(name, "legacy")] = (forward, backward)
    for key in workloads:
        validation.setdefault(key, {}).update({"mass": mass_info})
    return workloads, validation


def benchmark_cases(args, cases, checkpoint=None):
    import torch
    device = torch.device(f"cuda:{args.device}")
    # Existing entry points use the default stream. Keep comparisons on it;
    # the new fused entry points also accept the current PyTorch stream explicitly.
    rows = []
    forward_context = torch.enable_grad if args.forward_mode == "training" else torch.no_grad
    with GPUQuietGuard(args.device) as guard:
        torch.cuda.set_device(device)
        torch.manual_seed(args.seed)
        from sphops.sph import sph_cuda
        extension_path = Path(sph_cuda.__file__).resolve()
        environment = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(device),
                       "extension": str(extension_path),
                       "extension_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
                       "forward_mode": args.forward_mode,
                       "float32_matmul_precision": torch.get_float32_matmul_precision(),
                       "timing": "CUDA-event wrapper latency; kernel metrics require NCU"}
        for case in cases:
            for cap in args.block_caps:
                workloads, validation = make_workloads(torch, case, cap, args.channels,
                                                       args.hidden_dims, args.operations, args.variants, device,
                                                       mass_range=args.mass_range)
                if args.validate_only:
                    torch.cuda.synchronize()
                    guard.check()
                    for operation, variant in workloads:
                        rows.append({"case": case.name, "family": case.family,
                                     "block_cap": cap, "operation": operation, "variant": variant,
                                     "validation": validation.get((operation, variant), {"reference_variant": variant})})
                    del workloads
                    if checkpoint is not None:
                        checkpoint(rows, environment)
                    print(f"Validated {case.name}/cap={cap}")
                    continue
                if args.profile_only:
                    if len(workloads) != 1:
                        raise ValueError("--profile-only requires one case, block cap, operation, and variant")
                    (operation, variant), (forward, backward) = next(iter(workloads.items()))
                    for _ in range(args.warmup):
                        with forward_context():
                            forward()
                        backward()
                    torch.cuda.synchronize()
                    guard.check()
                    function = forward if args.profile_pass == "forward" else backward
                    # Autograd launches CUDA work on worker threads. A process
                    # start/end range includes those launches; a main-thread
                    # push/pop range would silently profile no backward kernels.
                    range_id = torch.cuda.nvtx.range_start("npa_profile")
                    try:
                        if args.profile_pass == "forward":
                            with forward_context():
                                function()
                        else:
                            function()
                        torch.cuda.synchronize()
                    finally:
                        torch.cuda.nvtx.range_end(range_id)
                    guard.check()
                    print(f"Profiled {case.name}/{operation}/{variant}/{args.profile_pass}")
                    continue
                samples, telemetry = defaultdict(list), defaultdict(list)
                for round_index in range(args.rounds):
                    keys = list(workloads)
                    if round_index % 2:
                        keys.reverse()
                    for operation, variant in keys:
                        forward, backward = workloads[(operation, variant)]
                        for direction, function in (("forward", forward), ("backward", backward)):
                            key = (operation, variant, direction)
                            if direction == "forward":
                                with forward_context():
                                    latency, info = event_latency(torch, function, args.iterations, args.warmup, guard)
                            else:
                                latency, info = event_latency(torch, function, args.iterations, args.warmup, guard)
                            samples[key].append(latency)
                            telemetry[key].append(info)
                for (operation, variant, direction), times in samples.items():
                    row = {"case": case.name, "family": case.family, "block_cap": cap,
                           "operation": operation, "variant": variant, "pass": direction,
                           **summarize(times), "telemetry": telemetry[(operation, variant, direction)],
                           "validation": validation.get((operation, variant), {"reference_variant": variant})}
                    rows.append(row)
                    print(f"{case.name:20} cap={cap:3} {operation:16} {variant:6} {direction:8} {row['median_ms']:.4f} ms")
                del workloads
                guard.check()
                if checkpoint is not None:
                    checkpoint(rows, environment)
    return rows, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="inspect CPU fixtures without any GPU use")
    parser.add_argument("--validate-only", action="store_true", help="check full-size outputs/adjoints without timing")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("benchmark_targets.yaml"))
    parser.add_argument("--case", nargs="+", dest="cases")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--channels", type=int, default=16,
                        help="logical state channels; fused supports 1..32 with automatic 4/8/16/32 padding")
    parser.add_argument("--hidden-dims", type=int, default=128)
    parser.add_argument("--mass-range", type=float, nargs=2, default=[1.0, 1.0], metavar=("MIN", "MAX"),
                        help="fixed masses: uniform [MIN,MAX), or a constant if MIN=MAX; same across variants and C")
    parser.add_argument("--block-caps", type=int, nargs="+", default=[32])
    parser.add_argument("--operations", nargs="+", default=["perception", "npa"],
                        choices=("perception", "npa", "density", "blur", "gradient", "density_gradient", "moment_matrix"))
    parser.add_argument("--variants", nargs="+", choices=("legacy", "fused"), default=["legacy", "fused"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--forward-mode", choices=("training", "inference"), default="training",
                        help="training includes autograd and cached backward data; inference disables grad")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/sph_benchmark.json"))
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-pass", choices=("forward", "backward"), default="forward")
    args = parser.parse_args()
    if min(args.channels, args.rounds, args.iterations, args.warmup, *args.block_caps) < 1:
        parser.error("channels, rounds, iterations, warmup, and block caps must be positive")
    if args.channels > 32 and "fused" in args.variants:
        parser.error("the fused variant supports at most 32 logical state channels")
    if not all(np.isfinite(args.mass_range)) or not 0 < args.mass_range[0] <= args.mass_range[1]:
        parser.error("--mass-range requires finite positive MIN <= MAX")
    if args.profile_only and (not args.cases or len(args.cases) != 1 or len(args.block_caps) != 1
                             or len(args.operations) != 1 or len(args.variants) != 1):
        parser.error("--profile-only requires exactly one --case, block cap, operation, and variant")
    if args.profile_only and args.validate_only:
        parser.error("--profile-only and --validate-only are mutually exclusive")
    cases = build_cases(args.manifest, batch_size=args.batch_size, num_particles=args.particles,
                        seed=args.seed, selected=args.cases)
    corpus = [{**case.metadata(), **neighbor_statistics(case)} for case in cases]
    for info in corpus:
        print(f"{info['name']:20} {info['shape']} neighbors={info['neighbors_mean']:.1f} "
              f"p95={info['neighbors_p95']:.0f} cells={info['occupied_cells']}")
    if args.prepare_only:
        return
    def save_results(rows, environment, complete):
        result = {"complete": complete,
                  "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "environment": environment, "corpus": corpus,
                  "validations" if args.validate_only else "measurements": rows,
                  "family_averages": [] if args.validate_only else aggregate(rows),
                  "comparisons": [] if args.validate_only else compare_variants(rows)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    rows, environment = benchmark_cases(args, cases,
                                        checkpoint=lambda rows, environment: save_results(rows, environment, False))
    # A caught numerical-comparison exception can keep its traceback/autograd
    # owners in a cycle. Collect after all workloads leave scope, before
    # nanobind's interpreter-shutdown reference diagnostics.
    gc.collect()
    if rows:
        save_results(rows, environment, True)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
