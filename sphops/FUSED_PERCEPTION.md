# Fused SPH perception

NPA enables `fused_perception=True` by default: `NPA.perceive` then builds the
hash grid, bins particles, and computes the perception features with fused
CUDA kernels (`sphops.fused_perception`). Setting it to `False`, or a workload
outside the supported configuration, selects the standalone SPH operators
(`grid.instantiate`, `sphops.bin_particles`, `sphops.perception`). Rebuild the
CUDA extension to make the fused kernels available; an extension without them
uses the standalone operators. Checkpoint keys and MLP dimensions are unchanged.

## Python API

```python
z, x_bin, s_bin, snapshot = sphops.fused_perception(
    x, s, grid, eps0, mass=None, perceive_state=True, density_grad=True,
    log_norm_grad=True, log_norm_density_grad=True)
```

`x`, `s`, and `mass` are in the original particle order, and `grid` is a
`sphops.HashGrid`. The result equals, up to float32 summation order:

```python
snapshot = grid.instantiate(x)
x_bin, s_bin = sphops.bin_particles(x, s, snapshot)
z = sphops.perception(x_bin.detach(), s_bin, snapshot, eps0)  # s_bin.detach() if not perceive_state
```

The `FusedPerception` autograd function makes one native call per direction.
Forward runs grid construction, binning, density, and features on the current
stream, keeping the snapshot in one int32 buffer and the state saved for
backward in one float workspace. Backward computes the perception adjoint and
adds the binned-state cotangent while unbinning, which is the sum autograd
would form for the two uses of `s_bin`. Positions receive gradients through
binning only. The returned `snapshot` is a regular `HashGridSnapshot`, so it
works with `bin_particles`, `unbin_particles`, and the standalone operators.
Compared with separate grid, binning, and perception calls, this saves about
125 us of host time per training step forward and 200 us forward+backward
on the test machine.
`sphops.fused_perception_supported(x, s, grid, mass)` reports whether inputs
are supported.

`sphops.perception(x_bin, s_bin, snapshot, eps0, ...)` composes the standalone
operators on binned particles. It is also differentiable with respect to
positions; NPA uses it when `stopgrad_pos=False`.

## Supported configurations

The fused path supports CUDA float32 positions and states, 2D or 3D, grid
mode, clamped or periodic boundaries, and 1-32 state channels. NPA uses it
when perception positions are detached (`stopgrad_pos=True`, the default) and
state-gradient perception is enabled. The kernels have a state-only backward:
they do not differentiate positions, density, or masses. Other modes, more
than 32 channels, position gradients, and disabled state-gradient perception
use the standalone operators, subject to their normal requirements.

The kernels are specialized for 4, 8, 16, and 32 channels. Other counts are
zero-padded to the next native size; each output field is sliced separately
and repacked to the logical width, and `s_bin` is returned as a view of the
padded binned state. Padding is differentiable, and the model never changes
its state or MLP dimensions. Native counts avoid this overhead. Padding can
reduce the performance benefit, particularly for 17-31 channels padded to 32.

Positions are `[B,N,D]`, so 3D positions are `[B,N,3]`. Earlier versions
padded them to four floats. Storing them unpadded measured no slower, and
the padding had leaked into the model, pool, losses, and `wrap_positions`.

## Fixed particle masses

Supply `mass=` to `NPA.forward`, `NPA.perceive`, or `fused_perception` in the
same original particle order as the input positions/states. `sphops.perception`
accepts masses in snapshot-binned order. All accept `[B,N]` or `[B,N,1]`
tensors; masses must match the positions' device/dtype and must not require
gradients. Use finite, positive masses. Omitted masses are ones; the fused path
reads a persistent ones buffer with the same mass-aware kernels instead of
allocating and binning one every step.

The density stage computes `rho_i = sum_j mass_j * W_ij` and caches
`volume_i = mass_i / rho_i`. Subsequent forward/state-backward stages use those
volumes. The density-gradient feature is divided by each sample's total mass
(N for unit masses), so it does not depend on the number of particles or on a
uniform rescaling of the masses. With explicit masses, the fused forward sums
them per sample in the original particle order (a fixed-order block reduction,
independent of the atomic binning order).

Rollout helpers return binned positions and states, whose particle order can
change at every step. `step_euler` does not accept masses: callers making
custom nonuniform-mass rollouts must pass `mass=` to the model and permute those
masses into the returned order before the next step. Reusing the original mass
array with already-binned positions would attach masses to the wrong particles.

## Optional compiled NPA step

`NPA(compile_step=True)` is an opt-in runtime optimization for `step_euler`:
PyTorch compiles the adaptation MLP, displacement normalization/statistics,
update-mask comparison, and Euler updates together. Native grid/perception and
random-number generation remain outside compilation. Random draws therefore
consume the same CUDA RNG sequence as the eager path. Checkpoint keys and
logical state dimensions are unchanged. Standalone `NPA.forward` and
`step_euler_trace` continue to use the eager adaptation path.

Compilation uses fixed shapes, default Inductor mode (no max-autotune sweep),
and no CUDA graph capture.
Keep batch/particle/channel counts and update settings stable to avoid
recompilation. It is disabled by default: warmup and additional saved tensors
can outweigh gains for short runs or memory-constrained workloads. See
[BENCHMARKS.md](BENCHMARKS.md) for the benchmark options.

## CUDA implementation

[`fused_perception.cuh`](fused_perception.cuh), in namespace `fused_perception`, implements
four stages; this is a fused perception pipeline, not one monolithic kernel.

| Stage | Work | Scheduling |
| --- | --- | --- |
| `density_forward` | Mass-weighted density, optional density gradient, cached volumes | Lane per particle; neighbor-tile warp teams |
| `forward` | Blur, raw state gradient, moment correction, scaling/log normalization, output packing | Lane per particle (two channel-group warps for 32 channels); neighbor-tile warp teams |
| `backward_prepare` | Normalization adjoint, radius scaling, direct/self state contribution, cotangent layout conversion | One thread per particle/channel |
| `backward` | Combined blur/raw-gradient neighbor adjoints | Lane per particle; neighbor-tile warp teams; no global atomics |

A grid block holds at most `max_particles_per_block` particles of one cell, so
all of its particles share one neighbor-candidate sequence: three merged-x row
spans in 2D, nine in 3D, plus a disjoint wrap chunk at periodic seams. Each
lane owns one particle. A neighbor's position, volume, and state (or
cotangents) are therefore warp-broadcast shared-memory reads (`float4` for
states and cotangents) with no shuffles, pair geometry and moment terms are computed once per pair, and
candidates outside the kernel support are skipped. 32 channels use two warps of
16 channels each to bound registers. Grid blocks larger than 32 particles are
processed in passes of 32.

One lane per particle alone leaves too few warps to hide latency (B=8, N=4096
is about 1,024 warps). Each grid block therefore runs `NEIGHBOR_WARPS` warp
teams. Each team takes every `NEIGHBOR_WARPS`-th neighbor tile (`TileCursor`)
into warp-private shared buffers (only `__syncwarp` in the neighbor loop). A
warp has two buffers: the next tile is copied with `cp.async` (global to
shared, committed as one group) while the current tile is processed, and
`cp.async.wait_group` plus `__syncwarp` make it visible; older targets than
sm_80 fall back to plain copies. Their per-lane partial sums are reduced in
fixed team order through shared memory that aliases the tiles (the staging may
extend beyond the tiles up to `REDUCE_STAGING_BYTES`, so the 2D forward
exchanges its 53 partials in one chunk). Results are deterministic, but the summation order
differs from one sequential sweep. Setting `NEIGHBOR_WARPS = 1` restores
the per-particle sequential order of the previous fused forward/backward
kernels (bitwise identical in 2D in our A/B checks).

The three neighbor kernels share `for_each_neighbor`. It assigns tiles to the
teams, holds the two warp barriers around each tile refill, and applies the
support test. The kernels differ only in what they stage per tile and what
they add per pair. The spans come from `GridConfig::neighbor_span` in
`utils.cuh`, built from `neighbor_row` and `row_x_range`, which the standalone
operators in `sph_cuda.cu` also use. The moment-matrix inverse
`safe_inverse_symmetric` is shared with `safe_inv_sym_mat`.

A grid block's particles are consecutive in binned order, so its output rows
form one contiguous range. Forward writes them through shared memory (again
aliasing the tiles): team 0 stores its lanes' rows with an odd row stride (no
bank conflicts), then all threads of the block copy the range with consecutive
threads on consecutive floats. Lane-per-particle stores would use 4 of every
32 bytes per sector. `gradient_values` is written the same way in a second
pass. Backward stages its `grad_state` row updates the same way. That measured
neutral there, since the neighbor traversal dominates, but keeps both kernels on
one output pattern.

The packed output is `[state(C), blur(C), state_gradient(C*D), density_gradient(D)]`;
the last field is omitted when disabled. Kernels use saved snapshot bin offsets
and block information, not the grid's mutable scratch counters. The active
block records are a contiguous prefix followed by zero-count sentinels, and the
neighbor kernels are persistent: a CUDA block loops over records `blockIdx.x +
k * gridDim.x` and stops at the first sentinel. The launch is at most eight
waves of resident blocks (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, cached
per kernel), so 2D launches are unchanged while the 3D ball launches 4,096
instead of 65,536 blocks.

Per pair, the forward computes the volume-weighted weights `v_j W` and
`v_j gradW` once, so each channel costs one fused multiply-add for the blur and
one per dimension for the gradient, in the exact `(s_j - s_i)` form. Periodic
displacements wrap as `d - L * rint(d / L)` (one rounding, one fma) instead of
`fmodf`; a displacement inside the box is returned exactly.

The moment-corrected forward gradient preserves the original straight-through
estimator:

```python
corrected = raw + (raw @ inverse_moment - raw).detach()
```

Consequently, backward applies the normalization Jacobian at the corrected
forward value but propagates through `raw`, without multiplying by the inverse
moment matrix. Degenerate moment matrices use the original determinant threshold
and identity fallback.

Log normalization preserves `log(1 + norm)`, including float32 rounding of
`1 + norm` and PyTorch's `1e-12` normalization clamp. The accurate small-norm
logarithm is intentional: replacing it with a fast intrinsic can significantly
change the state adjoint near zero.

The Python autograd context retains positions, volumes, normalization values,
weighted gradient sums, radius, and the snapshot. It need not save masses,
states, or moment matrices for backward. Barriers must remain block-uniform:
lanes without a particle still load tiles and reach every `__syncwarp` and
team-reduction `__syncthreads`, but skip pair work.

The extension is built without CUDA separable compilation (one translation
unit, no device-side launches). Relocatable device code made ptxas spill in
these kernels. Grid construction (`initialize`) keeps each particle's
arrival rank from the counting pass instead of running a second atomic pass,
clears only its counters (the record kernel writes the sentinel tail, other
outputs are fully written), runs both prefix sums as one scan over zipped
(count, blocks) pairs, and orders all work on the caller's stream: one memset
and five kernels. In the fused path the scatter kernel also copies positions,
states and masses into binned order (`grid_scatter_and_bin`), so a fused
forward is 8 launches and a backward 3 (prepare, gather, unbinning, which moves
rows as `float4`). Particle order within a cell remains arbitrary, as before.

The grid classes expose the fused path as `fused_perception_forward`,
`fused_perception_backward`, `fused_workspace_size`, and `fused_snapshot_words`
(`sph_cuda.cu`); `fused_workspace_layout` defines the saved-state layout.

The accompanying standalone-kernel fixes allocate/initialize CUB scan sentinels,
use snapshot-owned block bounds during later backward passes, and synchronize
shared gradient-buffer reads before reuse.

## Tuning knobs

The named constants in `fused_perception.cuh` hold the measured settings:

| Knob | Current | Swept | Notes |
| --- | ---: | --- | --- |
| `NEIGHBOR_WARPS` | 4 | 1, 2, 3, 4, 6 | Warp teams per grid block. 3-4 were best; 6 exceeds 48 KiB static shared memory for 3D C=32 backward |
| `FORWARD_RESIDENT_WARPS` | 16 | 16, 20 | `__launch_bounds__` register target; 20 spilled in the C=16 forward, and with 96 registers (no spills) the dense forward was 12% slower |
| `BACKWARD_RESIDENT_WARPS` | 32 | 24, 32 | Backward uses 64 registers; 24 (80 registers) was neutral overall (dense +3%, periodic −6%) |
| `DENSITY_RESIDENT_WARPS` | 32 | - | Density needs at most 55 registers |
| `NEIGHBOR_TILE` | 32 | - | Candidates per tile; halved (twice at most) until all of a warp's tile buffers fit `TILE_BYTES_PER_WARP` |
| `TILE_BYTES_PER_WARP` | 6144 | - | Bounds shared memory per resident block (e.g. 2D C=16 backward uses two 8-candidate buffers) |
| `FEATURE_TILE_BUFFERS`, `BACKWARD_TILE_BUFFERS`, `DENSITY_TILE_BUFFERS` | 2 | 1, 2 | Tile buffers per warp; 2 streams the next tile with `cp.async` while the current one is processed (backward 2D shape −10%, 3D −18%; density is neutral in 2D, −12% in 3D) |
| `REDUCE_STAGING_BYTES` | 21504 | 12288, 21504 | Team-reduction staging may grow beyond the tiles up to this size: one chunk (two barriers) for the 2D forward while four blocks stay resident |
| `MAX_CHANNELS_PER_THREAD` | 16 | - | 32 channels split across two warps |
| `max_particles_per_block` | 32 in configs | 16, 32, 64 | 16 leaves half of each lane-per-particle warp idle; prefer 32 or 64 |

Re-tune with the corpus workflow in [BENCHMARKS.md](BENCHMARKS.md): every sweep
should include forward and backward, dense disk seeds, uniform texture seeds,
and multiple image targets, then confirm with complete training and CUDA
instrumentation (including racecheck).
