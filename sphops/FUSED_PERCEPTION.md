# Fused SPH perception

NPA enables `fused_perception=True` by default. Setting it to `False` selects
the original SPH operators. The Python dispatcher automatically falls back for
workloads outside the fused implementation's supported configuration; rebuild
the CUDA extension to make the new kernels available. Checkpoint keys and MLP
dimensions are unchanged.

## Supported configurations

The fused path supports CUDA float32, contiguous particle positions/states,
2D or 3D, grid mode, clamped or periodic boundaries, and logical state-channel
counts 1–32. Perception positions must be detached (`stopgrad_pos=True`, the
NPA default), and state-gradient perception must be enabled. The kernel has a
state-only backward: it does not differentiate positions, density, or masses.
Other modes, C > 32, position gradients, and disabled state-gradient perception
use the original operations, subject to those operators' normal requirements.

Native CUDA channel counts are 4, 8, 16, and 32. PyTorch zero-pads other counts
to the next native size, slices each output field separately, and repacks the
original logical width. Padding/unpacking are differentiable; the model never
changes its state or MLP dimensions. Native counts avoid this overhead. Padding
can reduce the performance benefit, particularly for 17–31 channels padded to
32; use the opt-out when measurements favor the original operators.

3D positions use the existing `[B,N,4]` padded storage layout, but perception
gradient features have three spatial components.

## Fixed particle masses

Supply `mass=` to `NPA.forward` or `NPA.perceive` in the same original particle
order as the input positions/states. `sphops.perception` accepts masses in
snapshot-binned order. Both APIs accept `[B,N]` or `[B,N,1]` tensors; masses must
match the positions' device/dtype and must not require gradients. Use finite,
positive masses. Omitted masses become ones, using the same mass-aware kernels,
not a separate unit-mass specialization.

The density stage computes `rho_i = sum_j mass_j * W_ij` and caches
`volume_i = mass_i / rho_i`. Subsequent forward/state-backward stages use those
volumes. Density-gradient features retain the original division by particle
count N, not total mass.

Rollout helpers return binned positions and states, whose particle order can
change at every step. `step_euler` does not accept masses: callers making
custom nonuniform-mass rollouts must pass `mass=` to the model and permute those
masses into the returned order before the next step. Reusing the original mass
array with already-binned positions would attach masses to the wrong particles.

## CUDA implementation

[`perception.cuh`](perception.cuh), in namespace `fused_perception`, implements
four stages; this is a fused perception pipeline, not one monolithic kernel.

| Stage | Work | Scheduling |
| --- | --- | --- |
| `density_forward` | Mass-weighted density, optional density gradient, cached volumes | One cooperating warp per particle |
| `forward` | Blur, raw state gradient, moment correction, scaling/log normalization, output packing | One FEATURES-wide subwarp per particle; shared neighbor tiles |
| `backward_prepare` | Normalization adjoint, radius scaling, direct/self state contribution, cotangent layout conversion | One thread per particle/channel |
| `backward` | Combined blur/raw-gradient neighbor adjoints | Subwarp per particle; shared cotangent tiles; no global atomics |

The packed output is `[state(C), blur(C), state_gradient(C*D), density_gradient(D)]`;
the last field is omitted when disabled. Kernels use saved snapshot bin offsets
and block information, not the grid's mutable scratch counters. Zero-count
sentinel blocks safely handle the overprovisioned launch size.

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
states, or moment matrices for backward. Shared-memory barriers must remain
block-uniform: invalid local-particle groups still participate in tile loads,
shuffles, and barriers.

The accompanying standalone-kernel fixes allocate/initialize CUB scan sentinels,
use snapshot-owned block bounds during later backward passes, and synchronize
shared gradient-buffer reads before reuse. Package/standalone Python imports
reuse one native extension instance to avoid duplicate nanobind registration.

## Low-risk tuning candidates

No new tuning values were enabled during the documentation/default cleanup.
The named constants in `perception.cuh` currently retain the measured settings:

| Knob | Current | Initial sweep | Important tradeoff |
| --- | ---: | --- | --- |
| `BLOCK_THREADS` | 256 | 128, 256, 512 | Local particles per tile = threads / native FEATURES; larger blocks reuse tiles across more particles but can limit occupancy |
| `NEIGHBOR_TILE` | 64 | 32, 64 | Fewer barriers versus shared-memory footprint; 128 is only a candidate for smaller C |
| `DENSITY_THREADS` | 128 | 64, 128, 256 | Cooperating warps per grid block; full warps are required |
| `BACKWARD_PREPARE_THREADS` | 256 | 128, 256, 512 | Simple pointwise kernel; likely a smaller overall gain |
| `max_particles_per_block` | 32 in training configs | 16, 32, 64 | More/smaller grid blocks versus reuse and idle groups; also affects grid instantiation |

Prefer per-channel/per-pass choices over changing one global value for every
workload. With C=4 and a grid capacity of 32, the current 256-thread feature
kernel has 64 local groups, so half are inactive. At C=16/32, larger blocks can
instead reduce the number of repeated neighbor-tile traversals. These are
testable hypotheses, not measured speedups.

Backward shared storage is approximately
`NEIGHBOR_TILE * (sizeof(ParticlePosition<D>) + 4*(D+1)*C)` bytes. At C=32,
a tile of 128 needs 50,176 bytes in 2D or 67,584 bytes in 3D, exceeding the usual
48 KiB static shared-memory limit; do not blindly increase the tile for all
specializations. Forward and backward may benefit from different tile sizes.

A small code-level candidate is skipping feature arithmetic after broadcasting
zero support weights: currently out-of-radius candidate pairs still do channel
loads and multiply-adds. Measure the branch/divergence cost on both dense seeds
and sparse targets, and preserve uniform barrier/shuffle participation.

Use the validation/corpus/training workflows in [BENCHMARKS.md](BENCHMARKS.md)
before selecting defaults. Every sweep should include forward and backward,
dense disk seeds, uniform texture seeds, and multiple image targets; then confirm
the winner with complete training and CUDA instrumentation.
