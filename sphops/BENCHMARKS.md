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
| Lane-per-particle density/forward/backward with neighbor-tile warp teams | `fused_perception.cuh` | Dense disk C=16: 8.4x fewer warp instructions forward and 6.4x backward; NCU duration at base clocks 2.58 -> 0.38 ms forward, 2.46 -> 0.38 ms backward, 134 -> 99 us density |
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
training config now uses grid mode too; see the 3D section below.

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

## 3D growing config and particle-mode removal (2026-10-08)

`configs/growing-3dgs.yaml` used `mode: "particle"` (thread per particle,
Morton order, standalone operators only). On its ball seed (B=4, N=16384,
C=16, 32^3 grid, eps 0.1, cap 64; about 3.7 particles per occupied cell and
16.5 neighbors within eps), medians of 9 interleaved rounds on the RTX 4090:

| Workload (ms) | Particle mode | Grid, unfused | Grid, fused |
| --- | ---: | ---: | ---: |
| Perception forward / backward | 1.75 / 1.57 | 1.62 / 1.03 | 0.50 / 0.36 |
| NPA step forward / backward | 2.15 / 2.32 | 2.18 / 1.79 | 1.04 / 1.09 |
| 24-step rollout forward / backward | 52.2 / 53.2 | 52.6 / 40.8 | 25.2 / 26.2 |
| Rollout peak memory | 4290 MiB | 4312 MiB | 3517 MiB |

The rollout loss is the config's regularizers plus mean(decoded^2) in place of
the Gaussian-splat render (the lego data is not in the repository); rendering
costs the same in every mode. All three modes agree to about 3e-7 relative L2
in perception outputs and state adjoints.

Particle mode was then removed: the `KernelStrategy` template parameter, the
`vanilla_*` kernels, Morton hashing, and the `VHashGrid*` classes. `HashGrid`
still accepts `mode="grid"` for existing configs and rejects other modes with
a pointer to git tag `particle-mode-last` (commit c5cf88d), the last version
with particle mode. The 3D config now uses grid mode.

- Grid mode is unchanged: the 126 remaining kernels have identical SM89 SASS
  to the same kernels before the removal, and binning, every standalone
  operator (forward and backward), legacy perception and fused perception
  (unit and explicit masses) are bitwise identical in 2D/3D, both boundaries,
  power-of-two and other grid sizes, C = 5/16.
- `test_perception` (14 CPU + 15 CUDA), `test_pipeline` (6 CPU + 8 CUDA) and
  `testing/test.py` pass.

| Default build (`75 80 86 89`) | With particle mode | Grid only |
| --- | ---: | ---: |
| Compile time | 178.7 s | 99.9 s |
| Peak compiler memory | 2.94 GB | 1.89 GB |
| Kernels per architecture | 290 | 126 |
| SM89 cubin / PTX (uncompressed) | 15.9 / 11.3 MB | 6.5 / 4.9 MB |
| Extension size | 12.03 MB | 7.88 MB |

fatbinary compresses only large cubins by default. The smaller grid-only
cubins were stored uncompressed (a 30.0 MB extension), so `CMakeLists.txt` now
passes `-Xfatbin=-compress-all`; cubins were already stored compressed before.

Scripts and raw results are local in `results/fused_3d_ball_20261008/` and
`results/remove_particle_mode_20261008/` and are not source-controlled.

## Coalesced forward output (2026-10-08)

Nsight Compute flagged uncoalesced global accesses in the fused kernels. Changes:
forward stages its output and `gradient_values` rows in shared memory and writes
them coalesced, and loads each particle's own state as `float4`. Fused outputs and
adjoints stay bitwise identical (one particle per cell: binning, all standalone
operators, legacy and fused perception, 2D/3D, both boundaries, C = 5/16). With
dense and multi-pass blocks (caps 64/128, C = 5/16/32, unit and explicit masses),
the old/new difference equals the run-to-run binning-order noise (relative L2
≤ 1.2e-6). Racecheck, synccheck, memcheck and initcheck report no errors. The 3D
forward kernels with 16/32 channels gained a 16-byte stack frame (four spills,
outside the neighbor loop); shared memory is unchanged.

Kernel durations (NCU, locked base clocks, median of 5; B=8, N=4096, C=16, cap 32
in 2D; 3D is the `growing-3dgs.yaml` ball seed):

| Case | `forward` before → after | All four fused kernels |
| --- | ---: | ---: |
| emoji_lizard | 123.1 → 97.8 µs (1.26x) | 239.0 → 212.2 µs (1.13x) |
| texture_seed | 178.1 → 141.9 µs (1.26x) | 369.9 → 338.0 µs (1.09x) |
| texture_bubbly | 158.5 → 141.5 µs (1.12x) | 355.0 → 341.5 µs (1.04x) |
| growing_seed (dense disk) | 391.6 → 382.8 µs (1.02x) | 894.7 → 882.4 µs (1.01x) |
| ball_3d | 465.7 → 464.8 µs (1.00x) | 976.3 → 961.1 µs (1.02x) |

In the forward kernel, stores now use 26-27 of 32 bytes per sector (was 4). L2
throughput fell from 66% to 12% (lizard) and from 32% to 13% (3D ball). The 3D
and dense-seed kernels were not limited by these stores. Their remaining limits
are register-bound occupancy (128 registers, 33% theoretical) and idle lanes
(about 15 of 32 active), which come from the lane-per-particle design. Over the
12-target corpus, the end-to-end perception forward latency improved by 1.11x on
average. Backward and the NPA step were unchanged within run-to-run noise
(4-7%).

Backward stages its `grad_state` row updates the same way, so both kernels use
one output pattern. It measured neutral there (-3% to +4% by case) and stays
bitwise identical: the copy applies `fmaf(volume, sum, row)`, the rounding of the
former per-lane `+=`.

`backward_prepare` is unchanged: it already moves its minimum traffic at DRAM
bandwidth. It reads 13.2 MB for a 13.1 MB minimum in 2D and 36.4 MB for 35.1 MB
in 3D, at 0.8-0.9 TB/s. Its "uncoalesced" message describes per-instruction L1
sector use.

### Unpadded 3D positions

3D positions are now `[B,N,3]` everywhere (`ParticlePosition<3>` holds a
`float3`), instead of a `float4` with a zero fourth component. The padding had
leaked into `Pool`, both Euler steps, the compiled update, the 3D splat loss,
and `HashGrid.wrap_positions`, which returned three columns in 3D periodic mode.
It also put a zero column into `bound_regularizer`'s mean, which scaled that term
by 3/4 in 3D; the configured weight now applies as written.

- Bitwise A/B against the padded build, one particle per cell: 696/696 tensors
  identical. They cover binning, all standalone operators with position
  gradients, legacy and fused perception with gradients through binning, and an
  NPA step, in 2D/3D, both boundaries, C = 5/16, with unit and explicit masses.
  Every padded position tensor of the old build had an all-zero fourth column.
- Dense and multi-pass blocks: old/new difference ≤ 7e-7 relative L2, the same
  as the run-to-run binning-order noise.
- The standalone operators pass the PyTorch reference in 3D, forward and
  backward. Racecheck, synccheck, memcheck and initcheck report no errors.
- NCU kernel time over the whole fused path (binning included, 5 repeats): 2D
  cases are within ±3%. The 3D ball improves from 1010 to 989 µs; the fused
  kernels go from 940 to 919 µs. `apply_permutation` is the only slower kernel
  (13.1 → 15.7 µs): a 12-byte position copies as three scalar moves.
- Random draws shaped like positions now have three columns in 3D: position
  noise and the `gaussian`/`uniform` seeds. Seeded 3D runs therefore draw
  different numbers.

Scripts and reports are local in `results/padding_experiment_20261008/` and
`results/remove_particle_mode_20261008/`.

### Code cleanup

Duplicated code was merged, and `sph_cuda.cu`, `fused_perception.cuh` and
`utils.cuh` went from 4081 to 3387 lines:
- The neighbor spans are `GridConfig::neighbor_row`, `row_x_range` and
  `neighbor_span` in `utils.cuh`.
- The fused kernels walk tiles with one `for_each_neighbor`.
- One `safe_inverse_symmetric` serves the fused kernel and `safe_inv_sym_mat`,
  through a single templated kernel.
- The standalone operators use shared launch helpers (`OperatorArgs`,
  `launch_forward`, `launch_backward`), shape helpers (`per_particle`,
  `operator_args`) and a `RowSplit` index helper.
- The binning kernels share `copy_row` and `GridConfig::cell_hash`.
- `_init` sets the grid sizes, and the fused entry points use `snapshot_view`.
- Unused position operators and power helpers were removed.

Checks:
- Bitwise identical to the previous build: 696/696 tensors (2D/3D, both
  boundaries, C = 5/16, unit and explicit masses, all standalone forward and
  backward operators, legacy and fused perception, an NPA step).
- Dense and multi-pass blocks differ by at most 1e-6 relative L2, the
  run-to-run binning noise.
- The standalone operators pass the PyTorch reference in 2D and 3D.
- Racecheck, memcheck, initcheck and synccheck report no errors.
- NCU kernel time is unchanged within noise: standalone operators 1.00-1.03x,
  fused path 1.01-1.05x (2D lizard, texture seed, 3D ball, and more).

The standalone kernels compute each row's span inline from `neighbor_row` and
`row_x_range` instead of calling `neighbor_span`. In those block-wide kernels,
the call raised register counts (3D gradient backward from 48 to 54) and made
the 2D periodic moment-matrix forward 44% slower, although the arithmetic was
identical. The inline form measures the same as before.

## Second optimization round (2026-10-08, after the cleanup)

Profiling the cleaned-up kernels with Nsight Compute (stall reasons, SASS-level
instruction counts) pointed at four things: global-load and barrier stalls in
the sparse forward, `fmodf` in every periodic displacement, shared-load stalls
in the backward, and 65,536 launched CUDA blocks in 3D of which 73% only read
an empty record. The changes, all in `fused_perception.cuh`, `sph_cuda.cu` and
`utils.cuh`:

| Change | Effect |
| --- | --- |
| Two tile buffers per warp; tiles copied with `cp.async` (global to shared, one commit group per tile) while the previous tile is processed; backward tiles of 8 candidates so that two buffers fit the shared-memory budget | Backward: shape −10%, dense −8%, 3D −18%; forward: dense −5%, 3D −9% (locked clocks) |
| Persistent neighbor kernels: a CUDA block loops over block records with stride `gridDim.x` and stops at the first zero-count sentinel; launch = min(records, 8 waves of resident blocks) | 3D ball: forward −3%, backward −4%, density −4%; 2D unchanged (launch size unchanged) |
| Periodic wrap `d - L * rint(d / L)` instead of `fmodf` | Periodic cases: density −48%, forward −20%, backward −10% |
| Volume-weighted weights computed once per pair: 4 instead of 5 floating-point instructions per channel in 2D (exact `(s_j - s_i)` form kept) | Dense forward −3%, periodic forward −6% |
| Team reduction in one chunk (staging up to 21 KB), both output row sets staged behind one barrier, no barrier after the last particle chunk | Part of the forward gains above; 4 block barriers per grid block instead of 10 |
| Scatter fused with binning, sentinel tail written by the record kernel, one zipped scan for both prefix sums, `float4` unbinning | 12 → 8 launches per fused forward; unbinning 13 → 5 µs (2D), 27 → 9 µs (3D) |

Nsight Compute kernel durations at locked base clocks (one profiled call per
build; the atomic binning order differs between processes and moves the times
of identical code by up to about 5%), B=8, N=4096, C=16, cap 32 in 2D; 3D is
the `growing-3dgs.yaml` ball seed (B=4, N=16384, cap 64):

| Case | `forward` | `backward` | `density_forward` |
| --- | ---: | ---: | ---: |
| growing_seed (dense disk, clamped) | 379.2 → 352.6 µs | 393.7 → 353.0 µs | 97.9 → 102.7 µs |
| texture_seed (uniform, periodic) | 142.4 → 106.0 µs | 115.8 → 104.1 µs | 61.1 → 31.6 µs |
| emoji_lizard (shape, clamped) | 95.0 → 90.3 µs | 78.5 → 74.6 µs | 21.6 → 22.5 µs |
| ball_3d (clamped) | 466.0 → 395.4 µs | 328.3 → 258.1 µs | 131.7 → 115.1 µs |

Kernel time per `fused_perception` call (CUPTI sum of all kernels of the call,
including grid build, binning and unbinning; normal clocks; three interleaved
fresh-process rounds per build, C=16):

| Case | Forward | Backward |
| --- | ---: | ---: |
| growing_seed | 427-496 → 396-407 µs | 340-373 → 310-332 µs |
| texture_seed | 183 → 132 µs | 117 → 94 µs |
| emoji_lizard | 113-128 → 105 µs | 81 → 67 µs |
| emoji_fox | 92 → 86 µs | 64 → 52 µs |
| texture_bubbly | 182-209 → 130 µs | 118 → 92 µs |
| texture_banded | 166-175 → 119-137 µs | 104 → 84 µs |
| ball_3d | 526-585 → 447-466 µs | 321-344 → 245-257 µs |

Correctness:

- `test_perception` (14 CPU + 29 CUDA), `test_pipeline` (14) and the
  standalone `testing/test.py` suite pass. Compute Sanitizer memcheck,
  synccheck and initcheck report 0 errors; racecheck (fused, grid and binning
  kernels, all reports) 0 hazards.
- Against the build before this round, on the corpus plus 3D ball/cube cases
  (C = 5/16/32, unit and explicit masses, outputs and state adjoints compared
  in original particle order): relative L2 at most 3.3e-6, all finite. The
  largest differences are periodic cases, where the wrap no longer rounds
  `d + L/2`; one-particle-per-cell cases differ by at most 7.7e-7.
- On states from a 40-step rollout of the pretrained lizard model (B=2), the
  forward and the state adjoint of both the old and the new build agree with a
  float64 dense reference (`sphops/testing/torch_sph`) to 1.3e-7 (gradient
  features), 4.3e-7 (density gradient) and 3.5e-7 (adjoint) relative L2.

Matched 300-update graft training (`benchmark_training`, current configs, so
the emoji run uses `compile_step`), before → after this round:

| Run | Timed updates | Rollout forward | Backward | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| Emoji fox, run 1 | 20.05 → 19.86 s | 9.09 → 8.91 s | 8.57 → 8.53 s | 5.07 GiB both |
| Emoji fox, run 2 | 20.14 → 19.74 s | 9.13 → 8.79 s | 8.59 → 8.53 s | 5.07 GiB both |
| Texture bubbly | 14.42 → 14.17 s | 3.38 → 3.29 s | 7.99 → 7.84 s | 2.41 GiB both |

The kernel-level gains of 10-25% become 1.5-2% of a training update: the step
is host-launch bound and fused perception is a minor part of its GPU time (see
the previous section). Losses stay within run-to-run variation (fox last-25
mean 1.173 / 1.253 before, 1.169 / 1.237 after; bubbly 2.371 before, 2.362
after). Further end-to-end gains need fewer launches per step (CUDA graphs, or
fusing the MLP and Euler update with the perception call), not faster
perception kernels.

Rejected experiments, same measurement:

- Centering the tile states on the block's first particle's state (3 instead of
  4 floating-point instructions per channel): dense forward another −5%, but
  gradient features 1.7e-7 instead of 1.35e-7 relative L2 from the float64
  reference (cancellation of `s_i * sum(v gradW)`), so the exact form was kept.
- `FORWARD_RESIDENT_WARPS = 20` (96 registers, no spills, 20 warps/SM):
  dense forward +12%, periodic +13%.
- Balanced contiguous candidate ranges per team instead of round-robin tiles,
  plus the epilogue split over the four teams: fewer barrier stalls but +20%
  instructions (a second pass over the spans, partial tiles); forward +17%
  shape, +40% 3D. The SASS profile shows the epilogue at about 2% of the sparse
  forward, so there was little to parallelize.
- Software-pipelined inner loop (displacement of candidate j+1 issued before
  the pair work of j): 2D forward −3 to −6%, but backward +5 to +9%, density
  +5%, 3D forward +4%; not kept.
- `BACKWARD_RESIDENT_WARPS = 24` (80 registers): dense +3%, periodic −6%, 3D
  −3%; neutral, not kept.
- Greedy count-balanced tile assignment (each tile to the team with the fewest
  candidates): the per-team load counters indexed dynamically spilled to local
  memory; density up to 3.5x slower; not kept.
- Persistent grid of two waves instead of eight: sparse 2D forward +5% (the
  hardware scheduler balances 3,072 short blocks better than 1,024 looping
  ones).

Full Nsight Compute reports of the final kernels (`--set full`, locked base
clocks, one `fused_perception` forward + backward on the dense disk and on the
emoji_lizard target, installed extension rebuilt from these sources for all
four architectures) are in `sphops/ncu_reports/fused_round2_20261008/`, with
the profiling script next to them. Open them with
`ncu --import <report> --page details` or in the Nsight Compute UI.

Scripts and raw results of this round (A/B timing, numerical comparison,
trained-state validation, Nsight exports of the intermediate variants) are
local in `results/kernel_opt_20261008/` and in the session scratch directory;
they are not source-controlled.

## Third optimization round (2026-10-08): several lanes per particle in sparse blocks

The second-round profiles left one structural inefficiency: with one lane per
particle, a grid block with few particles (shape boundaries, the 3D ball's
cells of about 10 particles, the ragged last pass of a dense block) runs its
warps with most lanes idle, and the active lanes per instruction were 16-18 of
32 in the sparse cases. The pass loop of the three neighbor kernels now gives
each particle 2, 4 or 8 lanes when at most 16, 8 or 4 particles are left in
the block (`lane_share_shift`, uniform over the block), capped by the block's
candidate density (`lane_share_cap`: 2, 4 or 8 lanes need at least 4, 8 or 16
candidates per neighbor row span). The lanes of one particle take interleaved
candidates of every tile (lane r takes candidates r, r + share, ...), reduce
their partial sums with an xor-shuffle butterfly before the team reduction,
and the first lane of each particle owns the outputs (`slot` instead of `lane`
indexes the staged rows). Passes with one lane per particle keep the previous
stride-1 candidate loop; the strided loop is a second branch of
`for_each_neighbor`. Dense blocks are unchanged.

The candidate density comes from the block record: `assign_block_info` sums
the lengths of the cell's neighbor row spans once per cell (3 spans in 2D, 9
in 3D, plus the periodic wrap chunks) and stores the average per row in a new
`BlockInfo::candidates_per_row` field. `BlockInfo` is therefore 32 bytes;
the extension exports `BLOCK_INFO_WORDS = 8` and `sph.py` sizes the record
tensor and the saved snapshot from it (older extension builds without the
constant still load with 4 words). The record kernel grew from 1.6 to 3.2 µs
in 2D and from 2.6 to 9.8 µs in 3D per grid build.

Nsight Compute kernel durations at locked base clocks, medians of three
(before) and four (after) profiled calls per build (C=16; 2D cases B=8,
N=4096, cap 32; 3D is the `growing-3dgs.yaml` ball seed, B=4, N=16384, cap
64), second round → this round, with the executed warp instructions and the
average active lanes per instruction of the forward kernel:

| Case | `forward` | `backward` | `density_forward` | Forward instructions | Active lanes (fwd / bwd) |
| --- | ---: | ---: | ---: | ---: | ---: |
| growing_seed (dense disk, clamped) | 350.0 → 332.2 µs | 350.8 → 351.5 µs | 102.7 → 103.8 µs | 214.6 → 207.9 M (−3%) | 18.5 → 19.1 / 20.4 → 21.0 |
| texture_seed (uniform, periodic) | 107.0 → 104.0 µs | 105.0 → 90.3 µs | 34.3 → 29.9 µs | 64.3 → 53.7 M (−16%) | 15.7 → 19.1 / 17.7 → 21.1 |
| emoji_lizard (shape, clamped) | 93.0 → 79.2 µs | 72.4 → 71.3 µs | 22.2 → 19.5 µs | 43.7 → 35.6 M (−18%) | 15.6 → 20.0 / 17.2 → 21.6 |
| ball_3d (clamped) | 396.2 → 359.9 µs | 258.6 → 214.2 µs | 113.0 → 106.6 µs | 209.6 → 185.9 M (−11%) | 16.2 → 22.2 / 17.1 → 23.5 |

Backward and density instructions fell by the same 12-18% in the sparse cases
(3D backward 167.2 → 137.4 M). The durations fall less than the instructions
because the sparse kernels are latency bound (issue slot utilization 45-55%
in the forward): the per-pass costs (tile loads, barriers, team reduction,
epilogue) are untouched, and the periodic uniform forward is dominated by
them. The emoji backward varies by ±5% between profiled calls (binning
order), so its −1.5% is within noise; its instruction count fell 17%.

The same kernels over particle counts (locked base clocks, medians of three
calls, same grids as the corpus so the particles per cell grow with N; 2D
B=8, 3D B=4):

| Case | N | `forward` | `backward` | `density_forward` |
| --- | ---: | ---: | ---: | ---: |
| growing_seed (dense disk) | 2,048 | 130.5 → 128.0 µs (−2%) | 131.6 → 130.8 µs (−1%) | 36.8 → 36.3 µs (−1%) |
| growing_seed (dense disk) | 4,096 | 353.6 → 334.7 µs (−5%) | 363.8 → 351.4 µs (−3%) | 104.2 → 102.0 µs (−2%) |
| growing_seed (dense disk) | 8,192 | 1,156 → 1,099 µs (−5%) | 1,112 → 1,110 µs (0%) | 300.6 → 302.0 µs (0%) |
| growing_seed (dense disk) | 16,384 | 4,248 → 4,021 µs (−5%) | 3,880 → 3,975 µs (+2%) | 970.9 → 974.6 µs (0%) |
| growing_seed (dense disk) | 32,768 | 16,242 → 15,451 µs (−5%) | 14,722 → 15,049 µs (+2%) | 3,571 → 3,527 µs (−1%) |
| texture_seed (uniform, periodic) | 2,048 | 46.8 → 44.2 µs (−6%) | 34.0 → 33.6 µs (−1%) | 15.4 → 14.6 µs (−6%) |
| texture_seed (uniform, periodic) | 4,096 | 111.9 → 106.2 µs (−5%) | 106.5 → 94.0 µs (−12%) | 31.6 → 29.6 µs (−6%) |
| texture_seed (uniform, periodic) | 8,192 | 299.7 → 291.9 µs (−3%) | 255.6 → 241.6 µs (−5%) | 86.9 → 85.9 µs (−1%) |
| texture_seed (uniform, periodic) | 16,384 | 988.9 → 967.8 µs (−2%) | 831.9 → 777.0 µs (−7%) | 261.5 → 258.7 µs (−1%) |
| texture_seed (uniform, periodic) | 32,768 | 3,599 → 3,607 µs (0%) | 3,051 → 2,879 µs (−6%) | 920.9 → 919.2 µs (0%) |
| emoji_lizard (shape) | 2,048 | 43.5 → 36.5 µs (−16%) | 29.3 → 25.0 µs (−15%) | 10.8 → 10.2 µs (−6%) |
| emoji_lizard (shape) | 4,096 | 89.3 → 77.4 µs (−13%) | 69.0 → 67.1 µs (−3%) | 21.5 → 19.6 µs (−9%) |
| emoji_lizard (shape) | 8,192 | 206.2 → 189.2 µs (−8%) | 175.0 → 167.5 µs (−4%) | 52.1 → 50.5 µs (−3%) |
| emoji_lizard (shape) | 16,384 | 643.8 → 577.0 µs (−10%) | 555.8 → 537.4 µs (−3%) | 151.3 → 145.0 µs (−4%) |
| emoji_lizard (shape) | 32,768 | 2,243 → 2,092 µs (−7%) | 1,929 → 1,924 µs (0%) | 490.3 → 481.0 µs (−2%) |
| texture_bubbly (periodic) | 2,048 | 48.6 → 45.5 µs (−7%) | 46.0 → 33.8 µs (−27%) | 16.5 → 15.1 µs (−9%) |
| texture_bubbly (periodic) | 4,096 | 112.1 → 104.7 µs (−7%) | 103.8 → 89.9 µs (−13%) | 32.1 → 30.0 µs (−7%) |
| texture_bubbly (periodic) | 8,192 | 293.4 → 279.1 µs (−5%) | 258.8 → 231.7 µs (−10%) | 86.2 → 86.0 µs (0%) |
| texture_bubbly (periodic) | 16,384 | 937.1 → 912.1 µs (−3%) | 808.4 → 740.5 µs (−8%) | 259.3 → 253.6 µs (−2%) |
| texture_bubbly (periodic) | 32,768 | 3,363 → 3,314 µs (−1%) | 2,853 → 2,660 µs (−7%) | 884.0 → 884.0 µs (0%) |
| ball_3d | 4,096 | 166.6 → 175.7 µs (+5%) | 96.9 → 99.2 µs (+2%) | 57.0 → 60.0 µs (+5%) |
| ball_3d | 8,192 | 256.1 → 261.9 µs (+2%) | 155.4 → 148.3 µs (−5%) | 82.2 → 87.3 µs (+6%) |
| ball_3d | 16,384 | 385.6 → 349.9 µs (−9%) | 257.8 → 212.3 µs (−18%) | 110.7 → 107.6 µs (−3%) |
| ball_3d | 32,768 | 606.1 → 481.5 µs (−21%) | 459.0 → 347.7 µs (−24%) | 153.6 → 136.8 µs (−11%) |
| ball_3d | 65,536 | 1,090 → 975.3 µs (−10%) | 851.3 → 724.6 µs (−15%) | 250.6 → 243.8 µs (−3%) |

The gains hold at every count and shrink as the cells fill up (at N=32,768
on the 16 × 16 grids most blocks hold 32 particles, which share nothing). In
3D the ball keeps its radius, so N sets the particles per cell: about 1 at
4,096, 13 at 16,384 (the configured case), 50 at 65,536. At 1 particle per
cell a tile holds about 3 candidates, which is where the cap matters: without
it (lanes shared by the particle count alone) the 3D forward was +21% at
N=4,096 and +13% at 8,192 (the butterfly of 73 partials per step with nothing
to split), with it +5% and +2%; above that the capped and the uncapped
versions are the same. The remaining few percent at the smallest counts are
the fixed cost of the sharing machinery (record load, lane arithmetic, the
loop branch) in blocks with little work.

Kernel time per `fused_perception` call (CUPTI sum of all kernels of the
call, including grid build, binning and unbinning; normal clocks; two interleaved fresh-process rounds per build, C=16; the ranges are the two rounds):

| Case | Forward | Backward |
| --- | ---: | ---: |
| growing_seed | 397-428 → 380 µs | 308-410 → 309-313 µs |
| texture_seed | 132 → 125-126 µs | 94 → 80-82 µs |
| emoji_lizard | 105 → 92-94 µs | 67 → 61-62 µs |
| emoji_fox | 85-86 → 79-81 µs | 52-53 → 49-50 µs |
| texture_bubbly | 130-147 → 124-126 µs | 92-93 → 80-83 µs |
| texture_banded | 119-151 → 117-120 µs | 84-85 → 74-76 µs |
| ball_3d | 447-504 → 408-416 µs | 242-325 → 198-202 µs |

Correctness:

- `test_perception` (14 CPU + 29 CUDA), `test_pipeline` (14) and the
  standalone `testing/test.py` suite pass; Compute Sanitizer memcheck,
  synccheck and initcheck 0 errors, racecheck (fused, grid and binning
  kernels) 0 hazards. The shuffle reduction is deterministic (same tree in
  every lane), so results stay bitwise reproducible for a given binning.
- Against the pre-round-2 build on the corpus plus the 3D and
  one-particle-per-cell cases (C = 5/16/32, unit and explicit masses):
  relative L2 at most 3.3e-6, unchanged from the second round (the
  differences are the second round's periodic wrap).
- Trained lizard rollout states (B=2, 40 steps) against the float64 dense
  reference: 1.3e-7 (gradient features), 4.1e-7 (density gradient) and 3.4e-7 (adjoint) relative L2, the same as before.
- `test_tiny_state_gradients_preserve_normalization_backward_accuracy` fails
  in some runs (3 of 5 full-suite runs with an intermediate build, 0 to 8 of
  40 isolated repetitions with the final one; one element of 3072 at state
  scale 1e-4, channels 16, always sample 1, particle 49, channel 4) and
  passes in others; the previous build showed the same in full-suite runs.
  Which runs fail depends on the atomic binning order. Against a float64
  dense reference (`tiny_diag.py`) the element is 0.81844; the fused adjoint
  gives 0.81764 (8.0e-4 off) and the test's float32 legacy reference 0.81694
  (1.5e-3 off), so the tolerance breach is the reference's float32
  cancellation in the normalization Jacobian at tiny gradient norms, and the
  fused kernel is the closer of the two. Normwise, both builds and the
  reference sit at the same 3.4-3.9e-4 relative distance from float64 at
  this scale (2.7e-7 at scale 1), with zero breaches at scales 1e-5 and 1.

Matched 300-update graft training (`benchmark_training`), second round → this
round: emoji fox 20.13 / 19.87 → 19.87 s (rollout forward 9.01 / 8.93 → 8.91 s, backward 8.61 / 8.44 → 8.51 s), texture bubbly 14.33 → 14.26 s; peak memory (5.07 / 2.41 GiB) and last-25 losses (fox 1.186 / 1.188 → 1.255, bubbly 2.350 → 2.352, within run-to-run variation) unchanged. As in the second round, the step is host-launch bound,
so kernel gains of 5-20% do not reach the training time.

Variants measured and not kept (same locked-clock method):

- Strided candidate loop in every pass (`j = sub; j < count; j += share`,
  also with an explicit trip count): −8 to −11% on the sparse 2D forward and
  backward, but the variable stride costs 2-3 integer instructions per
  candidate, which the dense density kernel (about 20 instructions per
  candidate) paid as +15%, and the dense backward +4%. The two-branch loop
  keeps the dense cases at the previous instruction counts.
- Contiguous candidate ranges per lane (lane r takes candidates
  [r·chunk, (r+1)·chunk)): stride-1 loop everywhere, but the lanes of one
  particle then read tile rows a multiple of 32 banks apart: 3x the shared
  bank conflicts (texture_seed backward 1.6 M versus 0.5 M), backward +8%
  sparse 2D.
- Padding the staged payload rows to an odd number of `float4` (forward
  states 16 → 20 floats, backward cotangents 48 → 52 in 2D, 64 → 68 in 3D):
  removes the remaining conflicts (3D backward 7.6 M → 31 k, shared
  wavefronts −50%) but no kernel is faster, and the larger tiles cost
  occupancy (2D backward 49% → 43%, emoji backward +4%). The conflicts are
  not on the critical path.
- Lanes shared by the particle count alone (no density cap): the particle
  sweep above, 3D forward +21% at N=4,096.
- Candidate density summed inside the neighbor kernels (every warp walks the
  9 spans of its block before the pass loop): the same 3D small-count cases
  +23% forward and +29% density, as expensive as the regression it removes.
- The cell's own particle count as the density proxy (free): fixes 3D, but
  the boundary cells of the dense disk have few particles and full
  neighbor rows, so it caps them wrongly: dense forward −1 to −3% instead of
  −5%. The record field costs nothing in the neighbor kernels.

The full Nsight Compute reports in `sphops/ncu_reports/fused_round2_20261008/`
were regenerated from these kernels (installed extension rebuilt for sm_75,
sm_80, sm_86 and sm_89; 124 s, 9.8 MB). Scripts and raw results (variant
builds A/B, bank-conflict counts, the particle sweep `scale_sweep_ncu.py` +
`parse_sweep.py` with its CSV exports under `sweep/`, the repeated tiny-state
check `tiny_diag.py`) are in `results/kernel_opt_20261008/`, not
source-controlled.
