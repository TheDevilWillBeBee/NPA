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
saved-snapshot lifetime, position-gradient fallback, multi-step NPA state
and parameter gradients, and the fused grid/bin/perceive path against the
standalone operators on its own snapshot (exact binning, position gradients,
and accumulated state gradients). `sphops.testing.test_pipeline` checks batched
splatting against the per-sample implementation.

```bash
python3 -m sphops.testing.test_perception
python3 -m sphops.testing.test_perception --gpu
python3 -m sphops.testing.test
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python3 -m sphops.testing.test_perception --gpu --instrumented
compute-sanitizer --tool synccheck --error-exitcode 1 \
  python3 -m sphops.testing.test_perception --gpu --instrumented
compute-sanitizer --tool initcheck --error-exitcode 1 \
  python3 -m sphops.testing.test_perception --gpu --instrumented
# Shared-memory hazards in the fused, grid, and binning kernels. Unrestricted
# racecheck needed >11 GB of host memory in a legacy multi-step test.
compute-sanitizer --tool racecheck --racecheck-report all --error-exitcode 1 \
  --kernel-name kns=fused_perception --kernel-name kns=grid_ \
  --kernel-name kns=apply_permutation --kernel-name kns=assign_block_info \
  python3 -m sphops.testing.test_perception --gpu --instrumented
```

Do not combine kernel filters with initcheck: memory written by unfiltered
kernels is then reported as uninitialized on later host copies.

The existing `sphops/testing/test.py` also benchmarks standalone operators; use its
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
python3 -m sphops.testing.benchmark --prepare-only
python3 -m sphops.testing.benchmark --validate-only --block-caps 16 32 64 \
  --output results/sph_validation.json
python3 -m sphops.testing.benchmark --channels 16 --operations perception npa \
  --block-caps 16 32 64 --iterations 64 --output results/sph_benchmark.json
python3 -m sphops.testing.benchmark --channels 5 --operations perception --block-caps 16 32
python3 -m sphops.testing.benchmark --channels 16 --mass-range 0.5 1.5 \
  --operations perception --block-caps 16 32
```

Edit [benchmark_targets.yaml](testing/benchmark_targets.yaml) for local asset paths.
Its machine-local cache/archive defaults are examples, not downloadable assets.
The runner never downloads images. Cases include a radius-0.2 growing seed,
uniform texture seed, intermediate growing disks, and four targets per image
family. Targets sample unique alpha-mask pixels at adaptive resolution; states
are synthetic, not trained hidden states. Positions/settings/masses are
fingerprinted, and target families average each target equally.

Default forward measurements include autograd/backward caching; backward is
timed separately on retained graphs. `--forward-mode inference` selects the
no-grad forward path. Padding, unpacking, and their adjoints are included.
Perception workloads start from unbinned particles, so both variants include
grid construction and binning (`fused_perception` versus `instantiate` +
`bin_particles` + `perception`); results recorded before 2026-10-07 timed
perception on pre-binned particles. Each variant bins independently, so
outputs are compared in original particle order.
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
  python3 -m sphops.testing.benchmark --profile-only --case growing_seed \
  --block-caps 32 --operations perception --variants fused --profile-pass backward
```

Use `--profile-pass forward` for forward. The process-wide NVTX range includes
autograd worker-thread launches. Open the exported `.ncu-rep` in Nsight Compute;
kernel sums under replay are not a substitute for the unprofiled wrapper or
training measurements.

## Complete graft-training benchmarks

[`benchmark_training.py`](testing/benchmark_training.py) follows the growing/texture
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
python3 -m sphops.testing.benchmark_training --variant unfused --epochs 300 \
  --graft data/pretrained/lizard.pth --target /path/to/fox.png \
  --output results/fox_graft_unfused
python3 -m sphops.testing.benchmark_training --variant fused --epochs 300 \
  --graft data/pretrained/lizard.pth --target /path/to/fox.png \
  --output results/fox_graft_fused
python3 -m sphops.testing.benchmark_training --config configs/texture.yaml \
  --graft data/pretrained/polka_dotted.pth --target data/transparent_textures/bubbly_0101.png \
  --variant unfused --epochs 300 --output results/bubbly_graft_unfused
python3 -m sphops.testing.benchmark_training --config configs/texture.yaml \
  --graft data/pretrained/polka_dotted.pth --target data/transparent_textures/bubbly_0101.png \
  --variant fused --epochs 300 --output results/bubbly_graft_fused
```

## RTX 4090 results (2026-10-06, previous fused kernels)

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

## Texture VGG batching and optional compiled update

`TextureLoss` passes color and density images through VGG in one batch, then
restores the original image/layer ordering; single-image losses keep one call.
VGG parameter gradients and normalization are unchanged. Different cuDNN batch
algorithms can change TF32 roundoff in image adjoints; production precision
settings are unchanged.

`NPA(compile_step=True)` optionally compiles the MLP and tensor-only Euler
update, leaving native grid/perception and random draws outside compilation.
It preserves checkpoint keys and RNG consumption, but introduces compilation
warmup and additional memory use. It defaults to `False`; enable it with
`npa.kwargs.compile_step` in YAML, or for controlled benchmarks:

```bash
python3 -m sphops.testing.benchmark_training --config configs/texture.yaml --variant fused \
  --compile-step --epochs 300 --graft data/pretrained/polka_dotted.pth \
  --target data/transparent_textures/bubbly_0101.png --output results/bubbly_compiled
```

Focused checks are available with `python3 -m sphops.testing.test_pipeline`
(CPU) and `--gpu` (including pretrained VGG image/parameter adjoints and real
2D/3D compiled updates). Measured effects are listed with the 2026-10-07
results below.

Loss splatting is batched (`utils.splatting.splat_particles_batch`): one pass
over all samples, without the per-sample loop's host synchronizations.

## RTX 4090 results (2026-10-07): lane-per-particle kernels, fused NPA path, batched splatting

The tables above describe the previous fused kernels. This section compares
that implementation (HEAD `0ebb5d0` kernels, built from a frozen source copy)
with the current one. Both are isolated SM89 Release builds selected with
`NPA_SPH_EXTENSION_PATH`, and both use the same Python code. The measurements
predate a later cleanup, which renamed the fused entry points, removed the
separate binned-input fused functions, made texture VGG batching
unconditional, and removed the benchmark options used to reproduce the
previous loss pipeline. Unless noted, kernel and training numbers below were
recorded before that cleanup.

| Change | Where | Effect |
| --- | --- | --- |
| Lane-per-particle density/forward/backward with neighbor-tile warp teams | `perception.cuh` | Dense disk C=16: 8.4x fewer warp instructions forward and 6.4x backward; NCU duration at base clocks 2.58 -> 0.38 ms forward, 2.46 -> 0.38 ms backward, 134 -> 99 us density |
| No CUDA separable compilation | `CMakeLists.txt` | Removes ptxas spills in the new kernels (up to 468 B per thread) |
| Grid build: arrival ranks, fewer memsets, stream-ordered | `sph_cuda.cu` | 9 instead of 14 launches per `initialize` |
| One autograd function for grid build, binning, and perception | `sph.py`, `models/npa.py` | About 200 us less host time per training step (forward+backward) |
| Batched splatting without per-sample loop or host syncs | `utils/splatting.py` | Removes 8 host syncs per loss forward and more in backward |

See [FUSED_PERCEPTION.md](FUSED_PERCEPTION.md) for the design.

### Correctness

- Restructuring: with `NEIGHBOR_WARPS = 1` (sequential summation), the
  feature forward was bitwise identical to the previous kernels in all 616
  configurations: 12 corpus cases plus dense/uniform 3D, C = 4/5/8/16/17/32,
  caps 16/32/64, unit/non-unit masses, and feature flags, on identical saved
  snapshots. Backward was identical in 2D; in 3D, 50 configurations differed
  by at most 8.7e-8 relative L2 (FMA contraction).
- Shipped configuration (4 teams, per-lane density sums): roundoff-level
  differences from the previous kernels. Output relative L2 is at most 9.1e-7
  (median 2.9e-7) and state adjoint at most 1.5e-6, all finite.
- Against the legacy operators on the full-size corpus (`--validate-only`,
  C = 5/16/32, unit and [0.5, 1.5) masses, caps 16/32/64): output relative L2
  at most 6.5e-7, state adjoint at most 1.8e-6 (limit 1e-5); all
  cached-normalization-point adjoint checks passed. The NPA-level
  parameter-gradient check at C=32 with non-unit masses fails for *both* the
  previous and current builds (3/3 repeats each, same magnitudes). A near-zero
  ReLU pre-activation (at most 1e-5) flips, and with random cotangents one
  particle carries about 0.5% of a weight-gradient norm, above the 0.1%
  normwise fallback limit. The perception-level checks of that configuration
  pass.
- The single grid/bin/perceive function was bitwise equal to the separate
  grid, binning, and (then also available) binned-input fused perception calls
  on the same snapshot: 288 subtests, including gradients with respect to
  positions and states.
- Batched splatting: on CPU, images and color gradients are identical to the
  per-sample implementation, and position gradients agree within 2.4e-7
  relative (different reduction order of separable tap distances). On CUDA,
  differences stay within `scatter_add`'s atomic-ordering roundoff.
- `test_perception` (13 CPU + 15 CUDA), `test_pipeline` (6 CPU + 8 CUDA), and
  the legacy `sphops/testing/test.py` suite pass. Compute Sanitizer memcheck,
  synccheck, and initcheck report 0 errors. Racecheck (all reports) shows 0
  hazards on the fused, grid, and binning kernels. Unrestricted racecheck was
  OOM-killed at 11.6 GB host memory in a legacy multi-step test, so run it
  with `--kernel-name kns=fused_perception` (plus `kns=grid_`,
  `kns=apply_permutation`, `kns=assign_block_info`) on machines with little RAM.
- Later change: the density-gradient feature is now divided by each sample's
  total mass instead of the particle count N (both fused and legacy paths).
  With unit masses (all current training configs) the fused outputs and state
  adjoints are bitwise unchanged (2D/3D, both boundaries, C = 5/16, one particle
  per cell so binning is deterministic). Explicit masses add one small
  per-sample reduction launch per forward; the measurements below predate it.

### Kernel benchmarks (B=8, N=4096, 2D, cap 32 unless noted)

Wrapper latency is `sphops.testing.benchmark --variants fused`, five interleaved
fresh-process rounds per build. It includes padding, Python autograd wrappers,
and launches. "Kernel time" sums CUPTI-measured GPU kernel durations per call
(`torch.profiler`, 20 calls), so it isolates the CUDA work in sparse cases
whose wrapper latency is host-bound. Family values average their targets
(4 emoji, 4 texture). "Equal-family" weights the disk seed, uniform texture
seed, emoji targets, and texture targets equally.

C=16 perception (training default), previous -> current:

| Family | Forward latency | Backward latency | Forward kernel time | Backward kernel time |
| --- | ---: | ---: | ---: | ---: |
| Disk seed (r=0.2) | 2.40 -> 0.41 ms (5.81x) | 2.09 -> 0.34 ms (6.13x) | 2.45 -> 0.43 ms (5.74x) | 2.15 -> 0.34 ms (6.34x) |
| Uniform texture seed | 0.70 -> 0.21 ms (3.34x) | 0.59 -> 0.12 ms (4.88x) | 0.70 -> 0.20 ms (3.41x) | 0.58 -> 0.10 ms (5.90x) |
| Emoji targets | 0.29 -> 0.11 ms (2.52x) | 0.27 -> 0.12 ms (2.33x) | 0.26 -> 0.10 ms (2.56x) | 0.24 -> 0.05 ms (4.59x) |
| Texture targets | 0.76 -> 0.19 ms (3.93x) | 0.65 -> 0.11 ms (5.64x) | 0.70 -> 0.20 ms (3.48x) | 0.61 -> 0.09 ms (6.49x) |
| Growing disks (r=0.4, 0.6) | 0.54 -> 0.16 ms (3.49x) | 0.50 -> 0.13 ms (3.77x) | 0.53 -> 0.15 ms (3.61x) | 0.49 -> 0.10 ms (5.10x) |
| **Equal-family** | **4.46x** | **5.19x** | **4.40x** | **6.13x** |

Equal-family speedups for other settings (latency forward / backward; kernel
time forward / backward):

| Setting | Latency | Kernel time |
| --- | --- | --- |
| C=4 | 3.30x / 3.56x | 3.83x / 6.01x |
| C=8 | 3.37x / 3.61x | - |
| C=32 | 4.98x / 6.56x | 5.00x / 6.65x |
| C=16, masses in [0.5, 1.5) | 4.49x / 4.97x | - |
| C=16, cap 16 | 2.84x / 3.30x | - |
| C=16, cap 64 | 4.31x / 5.27x | - |
| Full NPA step (perception + MLP + binning), C=16 | 2.86x / 2.10x | 3.31x / 2.93x |

With the current kernels, cap 32 has the lowest absolute C=16 latency (forward
0.23 ms vs 0.32 ms at cap 16 and 0.28 ms at cap 64). 3D (C=16, kernel time):
a dense ball seed improves 4.80x forward / 6.54x backward. A sparse uniform
cube with about 4 particles per cell improves only 1.99x / 2.63x (1.32x /
1.99x at C=32), because most lanes of each grid block's warps are idle. The 3D
training config uses particle mode, which does not use the fused path.

### Matched graft training (300 updates, B=8, N=4096, C=16, cap 32)

Previous: previous kernels and per-sample splatting. Current: current
kernels, single grid/bin/perceive function, and batched splatting; texture
VGG batching was still off in these runs. Each pair
had identical initial weights and pool, sampling indices, and rollout steps.
Emoji runs grafted `lizard.pth` (19,501 steps), texture runs
`polka_dotted.pth` (7,101 steps).

| Target | Previous | Current | Speedup | Peak allocated |
| --- | ---: | ---: | ---: | --- |
| Emoji fox | 38.18 s | 22.23 s | 1.72x | 4.44 -> 4.71 GiB |
| Emoji octopus | 36.32 s | 22.18 s | 1.64x | 4.28 -> 4.53 GiB |
| Texture bubbly | 24.64 s | 14.53 s | 1.70x | 2.47 -> 2.40 GiB |
| Texture banded | 24.96 s | 15.08 s | 1.66x | 2.61 -> 2.47 GiB |

Both families average 1.68x. Per epoch, emoji (fox) phases changed: rollout
40.1 -> 29.4 ms, loss 10.5 -> 6.2 ms, backward 75.3 -> 37.1 ms. Repeat runs
reproduced timings within 0.3%. Per-epoch losses differ at the level of
run-to-run variation: the previous build differs from itself by about 1%
median per epoch, because cell binning and `scatter_add` are atomic. Fox
final-25 / last-100 / all-300 mean losses were 1.170 / 1.233 / 1.628
(previous), 1.187 / 1.250 / 1.633 (previous, repeat), 1.245 / 1.292 / 1.681
(current), and 1.181 / 1.241 / 1.628 (current, repeat).

After the cleanup (fused kernels with byte-identical SASS; texture VGG batching
now always on), bubbly took 13.99 s and banded 14.67 s, 1.76x and 1.70x
faster than the previous implementation. Fox took 22.66-23.57 s in three runs
against 22.23-22.30 s before; those runs started at lower GPU clocks (2565 vs
2775-2790 MHz), and an in-process A/B attributes about 0.2% of rollout host
time to the cleanup.

Against the original unfused SPH operators (`--variant unfused`, previous
build), with matched runs and total time for 300 updates:

| Target | Unfused, per-sample splat | Unfused, batched | Current fused, per-sample | Current fused, batched |
| --- | ---: | ---: | ---: | ---: |
| Emoji fox | 54.91 s | 43.03 s | 35.73 s | 22.23 s |
| Emoji octopus | 52.63 s | 42.94 s | 31.62 s | 22.18 s |
| Texture bubbly | 33.37 s | 29.77 s | 19.57 s | 14.53 s |
| Texture banded | 33.58 s | 30.88 s | 20.05 s | 15.08 s |

Averaged over targets, the current fused path is 1.60x (emoji) / 1.69x
(texture) faster than unfused with per-sample splatting, and 1.94x / 2.05x
with batched splatting. Combined with batched splatting it is 2.42x / 2.26x
faster than the original unfused pipeline. The previous fused kernels gave
1.44x / 1.35x. Peak memory fell from 5.73-5.90 to 4.53-4.71 GiB (emoji) and
from 2.94-3.07 to 2.40-2.47 GiB (texture). Last-100-epoch losses varied by
target and run (octopus unfused, per-sample: 1.24 and 1.42 in two runs);
timings repeated within 0.1%.

The opt-in `--compile-step` reduced fox to 19.72 s (another 11%, peak
5.07 GiB). Compile step plus VGG batching (now always on) reduced bubbly to
13.52 s (another 7%, peak 2.62 GiB).

The training step is now largely host-launch bound. With current kernels,
fused perception is about 30% of an emoji epoch's GPU kernel time; the
remainder is mostly the MLP head, the Euler update, and their autograd backward.

### Tradeoffs

| | Previous | Current |
| --- | ---: | ---: |
| `sph_cuda.cu` compile, SM89 (2 runs) | 71.0 s | 80.1 s (+13%) |
| Peak compiler memory | 3.02 GB | 3.44 GB (+14%) |
| Extension size, SM89 | 5.13 MB | 4.73 MB |
| Device code, SM89 | sm_89 cubin only | sm_89 cubin (+9.6%) + compute_89 PTX |
| Default build (`75 80 86 89`) compile | 161.5 s | 181.7 s (+12.5%) |
| Default build extension size | 14.11 MB (4.52 MB fatbin) | 12.14 MB (9.88 MB fatbin) |

The relocatable (`-rdc`) build had dropped the PTX that CMake requests; it is
now embedded, so the extension can JIT on newer GPUs. Fused-perception SASS
grew about 50% (unrolled channel loops and team reductions). Per training
step, the fused path keeps the density gradient (B*N*D floats, 256 KB here)
until backward. Batched splatting materializes [B, N, P^2] tap tensors for the
whole batch (P^2 = 961 at the emoji loss resolution), raising emoji peak memory
by about 0.25 GiB; texture peak memory fell.

Isolated-build and A/B scripts used for these measurements are local in
`results/kernel_opt_20261007/` and are not source-controlled.
