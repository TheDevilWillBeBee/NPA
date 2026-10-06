# Fused perception validation and benchmarks

Run commands from the repository root. Detailed API, implementation, and tuning
constraints are in [FUSED_PERCEPTION.md](FUSED_PERCEPTION.md).
The benchmark utilities target Linux; their contention guard uses `fcntl`
advisory locking and `nvidia-smi`.

## Isolated build

This keeps the installed extension unchanged. For an RTX 4090, use SM89:

```bash
cmake -S sphops -B sphops/build/optimization -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DPython_EXECUTABLE="$(command -v python3)"
cmake --build sphops/build/optimization --parallel 2
export NPA_SPH_EXTENSION_PATH="$(realpath sphops/build/optimization/sph_cuda*.so)"
```

Select the appropriate architecture on other GPUs. To install normally, use the
repository build instructions. An older installed extension triggers the legacy
fallback until rebuilt. Changing the extension requires a fresh Python process.

## Correctness and instrumentation

The CPU suite checks feature packing, padding adjoints, mass validation, import
identity, checkpoint compatibility/default selection, and fixture generation.
CUDA tests also cover 2D/3D, both boundaries, native/padded channels, unequal
masses, normalization flags/tiny states, degenerate moments, periodic seams,
saved-snapshot lifetime, position-gradient fallback, and multi-step NPA state
and parameter gradients.

```bash
python3 -m sphops.test_perception
python3 -m sphops.test_perception --gpu
python3 sphops/test.py
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python3 -m sphops.test_perception --gpu --instrumented
compute-sanitizer --tool synccheck --error-exitcode 1 \
  python3 -m sphops.test_perception --gpu --instrumented
```

The existing `sphops/test.py` also benchmarks standalone operators; use its
`CUDAKernelValidator.run_validation()` for validation without timing.
The Python owner-lifetime assertion runs only without instrumentation, since
Compute Sanitizer can retain Python call frames. Instrumented shutdown may
report nanobind retained-reference warnings; those are distinct from CUDA
memory errors and must not occur in the normal owner-lifetime tests.

Adjoints are independently checked using legacy SPH and PyTorch autograd at
the fused kernel's cached normalization values. The unchanged rounded
`log(1 + norm)` can produce different Jacobians for roundoff-close tiny forward
values; cached-point checks remain pointwise, with an additional relative-L2
comparison against the original forward point. No moment-matrix multiplication
is added to the straight-through backward.

## Particle corpus and kernel benchmarks

```bash
python3 -m sphops.benchmark --prepare-only
python3 -m sphops.benchmark --validate-only --block-caps 16 32 64 \
  --output results/sph_validation.json
python3 -m sphops.benchmark --channels 16 --operations perception npa \
  --block-caps 16 32 64 --iterations 64 --output results/sph_benchmark.json
python3 -m sphops.benchmark --channels 5 --operations perception --block-caps 16 32
python3 -m sphops.benchmark --channels 16 --mass-range 0.5 1.5 \
  --operations perception --block-caps 16 32
```

Edit [benchmark_targets.yaml](benchmark_targets.yaml) for local asset paths.
Its machine-local cache/archive defaults are examples, not downloadable assets.
The runner never downloads images. Cases include a radius-0.2 growing seed,
uniform texture seed, intermediate growing disks, and four targets per image
family. Targets sample unique alpha-mask pixels at adaptive resolution; states
are synthetic, not trained hidden states. Positions/settings/masses are
fingerprinted, and target families average each target equally.

Default forward measurements include autograd/backward caching; backward is
timed separately on retained graphs. `--forward-mode inference` selects the
no-grad forward path. Padding, unpacking, and their adjoints are included.
`--operations density blur gradient density_gradient moment_matrix` selects
the standalone SPH operators, including full position/density/state backward.

Variants alternate order across timing rounds. The runner takes an advisory
GPU lock, monitors foreign compute processes, rejects contaminated runs, and
records clock/temperature/power. These are complete wrapper latencies, not
isolated CUDA-kernel active time or full optimizer-update time.

For NCU, select exactly one case, capacity, operation, variant, and pass:

```bash
mkdir -p results
ncu --clock-control none --cache-control none --nvtx --nvtx-include 'npa_profile' \
  --set basic --export results/sph_profile \
  python3 -m sphops.benchmark --profile-only --case growing_seed \
  --block-caps 32 --operations perception --variants fused --profile-pass backward
```

Use `--profile-pass forward` for forward. The process-wide NVTX range includes
autograd worker-thread launches. Open the exported `.ncu-rep` in Nsight Compute;
kernel sums under replay are not a substitute for the unprofiled wrapper or
training measurements.

## Complete graft-training benchmarks

[`benchmark_training.py`](benchmark_training.py) follows the growing/texture
training update recipe: pool sampling/damage, random-length rollout, loss and
regularizers, backward, gradient normalization, optimizer/scheduler, pool update.
It performs exactly the requested number of updates in one repetition. Warmup
updates are discarded before resetting the model, optimizer, pool, and RNG;
texture-loss warmup gradients are cleared too. Rendering, external logging,
stability evaluation, startup, warmup, and checkpoint I/O are excluded from
timed updates. Complete loop wall time additionally includes benchmark checks
and metric recording.

The texture run includes the existing VGG loss-network parameter backward;
the benchmark does not freeze VGG or change the loss. Its pretrained torchvision
weights should already be cached locally. Both runs verify the selected native
dispatch and finite losses, raw NPA gradients, states, and model parameters.
Configs, fingerprints, per-epoch timings/losses/sampling indices, and final model
checkpoints are saved in separate output directories; existing runs are not
overwritten. CUDA phase times include host launch gaps, not just GPU-active work.

Run both variants in fresh directories with the same config, graft, and target:

```bash
python3 -m sphops.benchmark_training --variant unfused --epochs 300 \
  --graft data/pretrained/lizard.pth --target /path/to/fox.png \
  --output results/fox_graft_unfused
python3 -m sphops.benchmark_training --variant fused --epochs 300 \
  --graft data/pretrained/lizard.pth --target /path/to/fox.png \
  --output results/fox_graft_fused
python3 -m sphops.benchmark_training --config configs/texture.yaml \
  --graft data/pretrained/polka_dotted.pth --target data/transparent_textures/bubbly_0101.png \
  --variant unfused --epochs 300 --output results/bubbly_graft_unfused
python3 -m sphops.benchmark_training --config configs/texture.yaml \
  --graft data/pretrained/polka_dotted.pth --target data/transparent_textures/bubbly_0101.png \
  --variant fused --epochs 300 --output results/bubbly_graft_fused
```

## RTX 4090 results (2026-10-06)

The unified fixed-mass implementation, with deterministic unequal masses in
`[0.5,1.5)`, was tested at B=8, N=4096 in 2D. Five interleaved rounds of 64
calls measured training-mode forward and separate state backward. Disk seed,
uniform texture seed, four emoji targets, and four texture targets received
equal family weight after averaging targets within family. All listed counts
are native; padding is not involved.

| C | Capacity 16 forward speedup | Capacity 16 backward speedup | Capacity 16 F+B speedup | Capacity 32 F+B speedup |
| --- | ---: | ---: | ---: | ---: |
| 4 | 1.57x | 1.01x | 1.30x | 1.47x |
| 8 | 1.53x | 1.07x | 1.31x | 1.59x |
| 16 | 1.65x | 1.34x | 1.50x | 1.32x |
| 32 | 1.83x | 1.12x | 1.47x | 1.29x |

Speedup = mean legacy latency / mean fused latency. The dense disk remains
the difficult case: capacity-16 combined speedups are 0.79x, 0.83x, 1.05x,
and 1.04x, respectively. These ratios compare variants at the same capacity;
they do not determine the fastest absolute capacity by themselves. Max output
relative-L2 error was 5.02e-7 and max state-adjoint error 1.28e-6. All 27
CPU/GPU tests, 56 legacy operation checks, and CUDA memcheck passed. Results
are local in `results/sph_fixed_mass_benchmark_20261006_9syJXu/`.

Matched 300-epoch graft training used C=16, hidden width 128, B=8, N=4096,
capacity 32, and the standard configs. Both variants had identical initial
weights/pool, sampling indices, and rollout-step sequences; the texture VGG
weights also matched. Particle trajectories are not bit-identical. Each result
is one matched seed/run, not a multi-seed convergence study.

| Measurement | Fox emoji unfused | Fox emoji fused | Bubbly texture unfused | Bubbly texture fused |
| --- | ---: | ---: | ---: | ---: |
| 300 timed training updates | 58.60 s | 39.33 s | 33.63 s | 25.18 s |
| Mean epoch | 195.3 ms | 131.1 ms | 112.1 ms | 83.9 ms |
| Forward rollouts, total | 23.65 s | 12.45 s | 12.13 s | 6.58 s |
| Loss forward, total | 3.36 s | 3.20 s | 4.59 s | 4.67 s |
| Full backward, total | 31.09 s | 23.21 s | 16.47 s | 13.49 s |
| Complete benchmark loop | 60.06 s | 40.50 s | 34.67 s | 26.15 s |
| Peak allocated memory | 5.90 GiB | 4.44 GiB | 2.94 GiB | 2.48 GiB |
| Final-25-epoch mean loss | 1.259 | 1.186 | 2.361 | 2.363 |

Timed-update speedups are 1.49x for emoji (32.9% less time) and 1.34x for texture
(25.1% less time). Emoji grafted `lizard.pth`, used a disk seed and 32–95 steps
per epoch (19,501 total); texture grafted `polka_dotted.pth`, used a uniform seed
and 16–31 steps (7,101 total). Both runs remained finite, with no foreign compute
processes detected. These historical measurements precede the namespace/comment
cleanup, which leaves the kernel math and launch values unchanged.

Local artifacts are `results/emoji_graft_training_20261006_E2WSr5/` and
`results/texture_graft_training_20261006_pb9eOS/`, each containing `comparison.json`,
`epochs.csv`, configs, logs, and final model checkpoints. Result files, compiled
extensions, profiler reports, and cached assets are not source-controlled.
