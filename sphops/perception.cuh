#pragma once

#include "utils.cuh"

// Fused SPH perception for NPA, not the standalone operators in sph_cuda.cu.
// Forward: mass-weighted density (+ optional density gradient), then one shared
// neighbor traversal for blur, state gradient, moment correction, normalization,
// and output packing. Backward: prepare cotangents, then gather blur + raw-gradient
// adjoints together. Positions/masses are fixed; only particle states differentiate.
// All buffers use snapshot-binned particle order. FEATURES is the native padded
// channel count (4/8/16/32), so feature loops/register arrays have static bounds;
// sph.py handles padding and fallback for unsupported workloads.
namespace fused_perception {

// Tuning knobs. Keep launch sites and these constants in sync: the feature
// kernels use BLOCK_THREADS for shared loads and BLOCK_THREADS / FEATURES local
// particles per tile. Larger neighbor tiles trade fewer barriers for more shared
// memory, especially in backward (NEIGHBOR_TILE * (DIM + 1) * FEATURES floats).
constexpr int BLOCK_THREADS = 256;
constexpr int NEIGHBOR_TILE = 64;
constexpr int DENSITY_THREADS = 128;
constexpr int BACKWARD_PREPARE_THREADS = 256;
static_assert(BLOCK_THREADS >= 32 && BLOCK_THREADS <= 1024 && BLOCK_THREADS % 32 == 0,
              "Feature blocks must contain complete warps for subwarp shuffles");
static_assert(DENSITY_THREADS >= 32 && DENSITY_THREADS <= 1024 && DENSITY_THREADS % 32 == 0,
              "Density blocks must contain complete cooperating warps");
static_assert(BACKWARD_PREPARE_THREADS > 0 && BACKWARD_PREPARE_THREADS <= 1024,
              "Backward preparation must use a valid CUDA block size");
static_assert(NEIGHBOR_TILE > 0, "Neighbor tiles must contain at least one particle");

// Cells are x-fastest in the bin offsets. Merge the three adjacent x cells into
// one contiguous span per y row (or y/z row in 3D); periodic seam cells need a
// second, disjoint wrap chunk. Spans contain candidates outside kernel support.
template <int DIM, BoundaryCondition BC>
__device__ __forceinline__ bool neighbor_span(
    const GridConfig<DIM, BC, KernelStrategy::GridBased> &config,
    const int *cell, unsigned int batch, int row, int wrap_chunk,
    const uint32_t *bin_offsets, uint32_t &start, uint32_t &end)
{
    int neighbor[DIM];
    neighbor[0] = cell[0];
    if constexpr (DIM == 2)
        neighbor[1] = cell[1] + row - 1;
    else {
        neighbor[1] = cell[1] + row / 3 - 1;
        neighbor[2] = cell[2] + row % 3 - 1;
    }
    if (!config.is_in_bound(neighbor))
        return false;
    config.apply_boundary(neighbor);

    int first_x, last_x;
    if (wrap_chunk != 0) {
        if (cell[0] == 0)
            first_x = last_x = config.grid_size[0] - 1;
        else if (cell[0] == config.grid_size[0] - 1)
            first_x = last_x = 0;
        else
            return false;
    } else {
        first_x = max(0, cell[0] - 1);
        last_x = min(config.grid_size[0] - 1, cell[0] + 1);
    }
    neighbor[0] = first_x;
    unsigned int first = config.cell2hash(neighbor);
    neighbor[0] = last_x;
    unsigned int last = config.cell2hash(neighbor);
    start = bin_offsets[batch * config.cell_count + first];
    end = bin_offsets[batch * config.cell_count + last + 1];
    return start != end;
}

template <int DIM, BoundaryCondition BC>
__device__ __forceinline__ void pair_weights(
    const GridConfig<DIM, BC, KernelStrategy::GridBased> &config,
    const ParticlePosition<DIM> &a, const ParticlePosition<DIM> &b,
    float &blur_weight, float *gradient_weight, ParticlePosition<DIM> &r)
{
    r = config.displacement(a, b);
    float d2 = dot(r, r);
    float h2 = config.eps * config.eps;
    blur_weight = 0.0f;
    #pragma unroll
    for (int d = 0; d < DIM; ++d)
        gradient_weight[d] = 0.0f;
    if (d2 >= h2)
        return;
    blur_weight = pow3(h2 - d2) * config.smoothing_coef;
    if (d2 > 0.0f) {
        float distance = sqrtf(d2);
        float magnitude = 3.0f * pow2(config.eps - distance)
                        / distance * config.spiky_coef;
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
            gradient_weight[d] = magnitude * r[d];
    }
}

__device__ __forceinline__ float warp_sum(float value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffU, value, offset);
    return value;
}

__device__ __forceinline__ float normalization_log(float norm)
{
    float rounded = __fadd_rn(1.0f, norm);
    // --use_fast_math lowers logf to __logf. Its absolute error is amplified
    // by division by a tiny norm in the normalizer and its adjoint. log1pf
    // keeps the accurate library path, but preserve PyTorch's rounded (1+norm)
    // rather than changing the operation to log1p(norm).
    return norm < 1.0f ? log1pf(__fsub_rn(rounded, 1.0f)) : logf(rounded);
}

template <int DIM>
__device__ __forceinline__ void log_normalize(float *value)
{
    float squared_norm = 0.0f;
    #pragma unroll
    for (int d = 0; d < DIM; ++d)
        squared_norm += value[d] * value[d];
    float norm = sqrtf(squared_norm);
    // Match log(1 + norm) * torch.nn.functional.normalize(..., eps=1e-12).
    float factor = normalization_log(norm) / fmaxf(norm, 1e-12f);
    #pragma unroll
    for (int d = 0; d < DIM; ++d)
        value[d] *= factor;
}

template <int DIM>
__device__ __forceinline__ void log_normalize_backward(
    const float *value, float *gradient)
{
    float squared_norm = 0.0f;
    #pragma unroll
    for (int d = 0; d < DIM; ++d)
        squared_norm += value[d] * value[d];
    float norm = sqrtf(squared_norm);
    if (norm == 0.0f) {
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
            gradient[d] = 0.0f;
        return;
    }
    float factor = normalization_log(norm) / fmaxf(norm, 1e-12f);
    float radial = norm > 1e-12f ? 1.0f / (1.0f + norm) - factor
                              : norm / (1e-12f * (1.0f + norm));
    float projection = 0.0f;
    #pragma unroll
    for (int d = 0; d < DIM; ++d)
        projection += gradient[d] * (value[d] / norm);
    #pragma unroll
    for (int d = 0; d < DIM; ++d)
        gradient[d] = factor * gradient[d]
                    + radial * projection * (value[d] / norm);
}

template <int DIM>
__device__ __forceinline__ void safe_inverse(float *matrix)
{
    // Match gradient_hybrid's symmetric inverse and determinant threshold.
    // Isolated/degenerate neighborhoods use identity, not an unstable inverse.
    if constexpr (DIM == 2) {
        float a = matrix[0], b = matrix[1], d = matrix[3];
        float det = a * d - b * b;
        if (fabsf(det) < 1e-3f) {
            matrix[0] = matrix[3] = 1.0f;
            matrix[1] = matrix[2] = 0.0f;
        } else {
            float inv_det = 1.0f / det;
            matrix[0] = d * inv_det;
            matrix[1] = matrix[2] = -b * inv_det;
            matrix[3] = a * inv_det;
        }
    } else {
        float a = matrix[0], b = matrix[1], c = matrix[2];
        float d = matrix[4], e = matrix[5], f = matrix[8];
        float t1 = d * f - e * e;
        float t2 = c * e - b * f;
        float t3 = b * e - c * d;
        float det = a * t1 + b * t2 + c * t3;
        if (fabsf(det) < 1e-3f) {
            #pragma unroll
            for (int k = 0; k < 9; ++k)
                matrix[k] = k % 4 == 0 ? 1.0f : 0.0f;
        } else {
            float inv_det = 1.0f / det;
            matrix[0] = t1 * inv_det;
            matrix[1] = matrix[3] = t2 * inv_det;
            matrix[2] = matrix[6] = t3 * inv_det;
            matrix[4] = (a * f - c * c) * inv_det;
            matrix[5] = matrix[7] = (b * c - a * e) * inv_det;
            matrix[8] = (a * d - b * b) * inv_det;
        }
    }
}

// Forward stage 1: one warp cooperates over a particle's neighbors to compute
// rho_i = sum_j m_j W_ij and optional density gradient. Cache volume_i = m_i/rho_i;
// subsequent fused stages need volume, not another density traversal or mass read.
template <int DIM, BoundaryCondition BC, bool DENSITY_GRAD>
__global__ void density_forward(
    const ParticlePosition<DIM> *__restrict__ positions,
    const float *__restrict__ mass,
    float *__restrict__ volume, float *__restrict__ density_gradient,
    const uint32_t *__restrict__ bin_offsets,
    const BlockInfo *__restrict__ block_info,
    GridConfig<DIM, BC, KernelStrategy::GridBased> config)
{
    // Snapshot-owned zero sentinels remain valid after later grid instantiations.
    BlockInfo info = block_info[blockIdx.x];
    if (info.particle_count == 0)
        return;
    int cell[DIM];
    config.hash2cell(info.cell_idx - info.batch_idx * config.cell_count, cell);
    unsigned int start = bin_offsets[info.cell_idx] + info.offset_in_cell;
    unsigned int lane = threadIdx.x & 31;
    unsigned int warp = threadIdx.x >> 5;
    constexpr int ROWS = DIM == 2 ? 3 : 9;
    constexpr int WRAP_CHUNKS = BC == BoundaryCondition::PERIODIC ? 2 : 1;
    for (unsigned int local = warp; local < info.particle_count; local += blockDim.x / 32) {
        unsigned int particle = start + local;
        ParticlePosition<DIM> position = positions[particle];
        float rho = 0.0f;
        float grad[DIM] = {0.0f};
        for (int row = 0; row < ROWS; ++row) {
            for (int wrap = 0; wrap < WRAP_CHUNKS; ++wrap) {
                unsigned int first, end;
                if (!neighbor_span(config, cell, info.batch_idx, row, wrap,
                                   bin_offsets, first, end))
                    continue;
                for (unsigned int neighbor = first + lane; neighbor < end; neighbor += 32) {
                    ParticlePosition<DIM> r = config.displacement(position, positions[neighbor]);
                    float d2 = dot(r, r);
                    float h2 = config.eps * config.eps;
                    if (d2 < h2) {
                        float neighbor_mass = mass[neighbor];
                        rho += neighbor_mass * pow3(h2 - d2);
                        if constexpr (DENSITY_GRAD) {
                            if (d2 > 0.0f) {
                                float distance = sqrtf(d2);
                                float magnitude = neighbor_mass * 3.0f * pow2(config.eps - distance) / distance;
                                #pragma unroll
                                for (int d = 0; d < DIM; ++d)
                                    grad[d] += magnitude * r[d];
                            }
                        }
                    }
                }
            }
        }
        rho = warp_sum(rho) * config.smoothing_coef;
        if (lane == 0)
            volume[particle] = mass[particle] / rho;
        if constexpr (DENSITY_GRAD) {
            #pragma unroll
            for (int d = 0; d < DIM; ++d) {
                float reduced = warp_sum(grad[d]);
                if (lane == 0)
                    density_gradient[particle * DIM + d] = reduced * config.spiky_coef;
            }
        }
    }
}

// Forward stage 2: a FEATURES-wide subwarp owns each particle, one lane per
// channel. Lane 0 computes geometry/moments once and broadcasts to its channels.
// Pack [state(C), blur(C), corrected-gradient(C*DIM), optional density-grad(DIM)].
template <int DIM, int FEATURES, BoundaryCondition BC>
__global__ void forward(
    const ParticlePosition<DIM> *__restrict__ positions,
    const float *__restrict__ state, const float *__restrict__ volume,
    const float *__restrict__ density_gradient, const float *__restrict__ eps0,
    float *__restrict__ perception, float *__restrict__ gradient_values,
    float *__restrict__ gradient_sum,
    const uint32_t *__restrict__ bin_offsets,
    const BlockInfo *__restrict__ block_info,
    GridConfig<DIM, BC, KernelStrategy::GridBased> config,
    bool perceive_density_grad, bool normalize_state_grad,
    bool normalize_density_grad, bool save_backward)
{
    static_assert(FEATURES == 4 || FEATURES == 8 || FEATURES == 16 || FEATURES == 32,
                  "Native feature counts must fit a power-of-two subwarp");
    BlockInfo info = block_info[blockIdx.x];
    if (info.particle_count == 0)
        return;
    __shared__ ParticlePosition<DIM> neighbor_pos[NEIGHBOR_TILE];
    __shared__ float neighbor_state[NEIGHBOR_TILE * FEATURES];
    __shared__ float neighbor_volume[NEIGHBOR_TILE];

    constexpr int LOCAL_TILE = BLOCK_THREADS / FEATURES;
    constexpr int ROWS = DIM == 2 ? 3 : 9;
    constexpr int WRAP_CHUNKS = BC == BoundaryCondition::PERIODIC ? 2 : 1;
    int cell[DIM];
    config.hash2cell(info.cell_idx - info.batch_idx * config.cell_count, cell);
    unsigned int start = bin_offsets[info.cell_idx] + info.offset_in_cell;
    unsigned int group = threadIdx.x / FEATURES;
    unsigned int channel = threadIdx.x % FEATURES;
    unsigned int output_dim = FEATURES * (2 + DIM) + (perceive_density_grad ? DIM : 0);
    float scale = config.eps / eps0[0];

    for (unsigned int offset = 0; offset < info.particle_count; offset += LOCAL_TILE) {
        // Invalid groups still participate in shuffles and shared-memory barriers;
        // never early-return individual lanes from this tiled traversal.
        bool valid = offset + group < info.particle_count;
        unsigned int particle = start + offset + group;
        ParticlePosition<DIM> position{};
        float local_state = 0.0f;
        if (valid) {
            position = positions[particle];
            local_state = state[particle * FEATURES + channel];
        }
        float blur = 0.0f;
        float gradient[DIM] = {0.0f};
        float matrix[DIM * DIM] = {0.0f};
        float weighted_sum[DIM] = {0.0f};

        for (int row = 0; row < ROWS; ++row) {
            for (int wrap = 0; wrap < WRAP_CHUNKS; ++wrap) {
                unsigned int first, end;
                if (!neighbor_span(config, cell, info.batch_idx, row, wrap,
                                   bin_offsets, first, end))
                    continue;
                for (unsigned int remote = first; remote < end; remote += NEIGHBOR_TILE) {
                    unsigned int count = min(unsigned(NEIGHBOR_TILE), end - remote);
                    for (unsigned int k = threadIdx.x; k < count; k += BLOCK_THREADS) {
                        neighbor_pos[k] = positions[remote + k];
                        neighbor_volume[k] = volume[remote + k];
                    }
                    for (unsigned int k = threadIdx.x; k < count * FEATURES; k += BLOCK_THREADS)
                        neighbor_state[k] = state[remote * FEATURES + k];
                    __syncthreads();  // All neighbor-tile loads must be visible.

                    for (unsigned int j = 0; j < count; ++j) {
                        float weight = 0.0f;
                        float grad_weight[DIM] = {0.0f};
                        if (valid && channel == 0) {
                            ParticlePosition<DIM> r;
                            pair_weights(config, position, neighbor_pos[j], weight, grad_weight, r);
                            float v = neighbor_volume[j];
                            #pragma unroll
                            for (int d = 0; d < DIM; ++d) {
                                if (save_backward)
                                    weighted_sum[d] += v * grad_weight[d];
                                #pragma unroll
                                for (int e = d; e < DIM; ++e)
                                    matrix[d * DIM + e] += v * grad_weight[d] * r[e];
                            }
                        }
                        weight = __shfl_sync(0xffffffffU, weight, 0, FEATURES);
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            grad_weight[d] = __shfl_sync(0xffffffffU, grad_weight[d], 0, FEATURES);
                        float v = neighbor_volume[j];
                        float remote_state = neighbor_state[j * FEATURES + channel];
                        blur += remote_state * (weight * v);
                        float difference = (remote_state - local_state) * v;
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            gradient[d] += difference * grad_weight[d];
                    }
                    __syncthreads();  // Finish tile reads before any lane refills it.
                }
            }
        }
        if (channel == 0) {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
                #pragma unroll
                for (int e = d + 1; e < DIM; ++e)
                    matrix[e * DIM + d] = matrix[d * DIM + e];
            safe_inverse<DIM>(matrix);
        }
        #pragma unroll
        for (int k = 0; k < DIM * DIM; ++k)
            matrix[k] = __shfl_sync(0xffffffffU, matrix[k], 0, FEATURES);

        // Moment correction changes the forward value only: gradient_hybrid
        // uses raw + (raw @ inverse - raw).detach(). Its adjoint is handled below.
        float corrected[DIM] = {0.0f};
        #pragma unroll
        for (int d = 0; d < DIM; ++d) {
            #pragma unroll
            for (int e = 0; e < DIM; ++e)
                corrected[d] += gradient[e] * matrix[e * DIM + d];
            corrected[d] *= scale;
            if (valid && save_backward && normalize_state_grad)
                gradient_values[(particle * FEATURES + channel) * DIM + d] = corrected[d];
        }
        if (normalize_state_grad)
            log_normalize<DIM>(corrected);
        if (valid) {
            unsigned int base = particle * output_dim;
            perception[base + channel] = local_state;
            perception[base + FEATURES + channel] = blur;
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
                perception[base + 2 * FEATURES + channel * DIM + d] = corrected[d];
            if (channel == 0) {
                if (save_backward) {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        gradient_sum[particle * DIM + d] = weighted_sum[d];
                }
                if (perceive_density_grad) {
                    float density_scale = 1.0f;
                    #pragma unroll
                    for (int d = 0; d < DIM + 1; ++d)
                        density_scale *= scale;
                    // Preserve NPA's divide-by-N convention, also for unequal masses.
                    density_scale /= config.particle_count;
                    float grad_rho[DIM];
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        grad_rho[d] = density_gradient[particle * DIM + d] * density_scale;
                    if (normalize_density_grad)
                        log_normalize<DIM>(grad_rho);
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        perception[base + FEATURES * (2 + DIM) + d] = grad_rho[d];
                }
            }
        }
    }
}

// Backward stage 1: undo log normalization and radius scaling once, write the
// direct-state and local/self-gradient contribution, then arrange blur + raw
// gradient cotangents as [particle, component, feature] for coalesced tile loads.
// The straight-through moment correction is NOT differentiated as raw @ inverse.
template <int DIM, int FEATURES>
__global__ void backward_prepare(
    const float *__restrict__ grad_perception,
    const float *__restrict__ gradient_values,
    const float *__restrict__ gradient_sum, const float *__restrict__ eps0,
    float *__restrict__ incoming, float *__restrict__ grad_state,
    unsigned int particles, unsigned int output_dim, float eps,
    bool normalize_state_grad)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particles * FEATURES)
        return;
    unsigned int particle = index / FEATURES;
    unsigned int channel = index % FEATURES;
    unsigned int base = particle * output_dim;
    float gradient[DIM], value[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; ++d) {
        gradient[d] = grad_perception[base + 2 * FEATURES + channel * DIM + d];
        if (normalize_state_grad)
            value[d] = gradient_values[index * DIM + d];
    }
    if (normalize_state_grad)
        log_normalize_backward<DIM>(value, gradient);
    float scale = eps / eps0[0];
    float local = grad_perception[base + channel];
    incoming[particle * (DIM + 1) * FEATURES + channel] =
        grad_perception[base + FEATURES + channel];
    #pragma unroll
    for (int d = 0; d < DIM; ++d) {
        // Straight-through moment correction: no multiplication by its inverse here.
        float raw_gradient = gradient[d] * scale;
        incoming[(particle * (DIM + 1) + d + 1) * FEATURES + channel] = raw_gradient;
        local -= raw_gradient * gradient_sum[particle * DIM + d];
    }
    grad_state[index] = local;
}

// Backward stage 2: gather both blur and raw-gradient neighbor contributions in
// one traversal. Each lane owns one final grad_state entry: no global atomics.
// volume_i multiplies incoming neighbor contributions; the local/self term was
// already written by backward_prepare. No position, density, or mass adjoints.
template <int DIM, int FEATURES, BoundaryCondition BC>
__global__ void backward(
    const ParticlePosition<DIM> *__restrict__ positions,
    const float *__restrict__ volume, const float *__restrict__ incoming,
    float *__restrict__ grad_state, const uint32_t *__restrict__ bin_offsets,
    const BlockInfo *__restrict__ block_info,
    GridConfig<DIM, BC, KernelStrategy::GridBased> config)
{
    static_assert(FEATURES == 4 || FEATURES == 8 || FEATURES == 16 || FEATURES == 32,
                  "Native feature counts must fit a power-of-two subwarp");
    BlockInfo info = block_info[blockIdx.x];
    if (info.particle_count == 0)
        return;
    __shared__ ParticlePosition<DIM> neighbor_pos[NEIGHBOR_TILE];
    __shared__ float neighbor_incoming[NEIGHBOR_TILE * (DIM + 1) * FEATURES];
    constexpr int LOCAL_TILE = BLOCK_THREADS / FEATURES;
    constexpr int ROWS = DIM == 2 ? 3 : 9;
    constexpr int WRAP_CHUNKS = BC == BoundaryCondition::PERIODIC ? 2 : 1;
    int cell[DIM];
    config.hash2cell(info.cell_idx - info.batch_idx * config.cell_count, cell);
    unsigned int start = bin_offsets[info.cell_idx] + info.offset_in_cell;
    unsigned int group = threadIdx.x / FEATURES;
    unsigned int channel = threadIdx.x % FEATURES;
    for (unsigned int offset = 0; offset < info.particle_count; offset += LOCAL_TILE) {
        bool valid = offset + group < info.particle_count;
        unsigned int particle = start + offset + group;
        ParticlePosition<DIM> position{};
        if (valid)
            position = positions[particle];
        float result = 0.0f;
        for (int row = 0; row < ROWS; ++row) {
            for (int wrap = 0; wrap < WRAP_CHUNKS; ++wrap) {
                unsigned int first, end;
                if (!neighbor_span(config, cell, info.batch_idx, row, wrap,
                                   bin_offsets, first, end))
                    continue;
                for (unsigned int remote = first; remote < end; remote += NEIGHBOR_TILE) {
                    unsigned int count = min(unsigned(NEIGHBOR_TILE), end - remote);
                    for (unsigned int k = threadIdx.x; k < count; k += BLOCK_THREADS)
                        neighbor_pos[k] = positions[remote + k];
                    for (unsigned int k = threadIdx.x; k < count * (DIM + 1) * FEATURES; k += BLOCK_THREADS)
                        neighbor_incoming[k] = incoming[remote * (DIM + 1) * FEATURES + k];
                    __syncthreads();
                    for (unsigned int j = 0; j < count; ++j) {
                        float weight = 0.0f;
                        float grad_weight[DIM] = {0.0f};
                        if (valid && channel == 0) {
                            ParticlePosition<DIM> r;
                            pair_weights(config, position, neighbor_pos[j], weight, grad_weight, r);
                        }
                        weight = __shfl_sync(0xffffffffU, weight, 0, FEATURES);
                        result += weight * neighbor_incoming[j * (DIM + 1) * FEATURES + channel];
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d) {
                            grad_weight[d] = __shfl_sync(0xffffffffU, grad_weight[d], 0, FEATURES);
                            // The gradient at j uses displacement j -> i, the negative
                            // of the forward displacement i -> j computed above.
                            result -= grad_weight[d] * neighbor_incoming[(j * (DIM + 1) + d + 1) * FEATURES + channel];
                        }
                    }
                    __syncthreads();
                }
            }
        }
        if (valid)
            grad_state[particle * FEATURES + channel] += volume[particle] * result;
    }
}

} // namespace fused_perception
