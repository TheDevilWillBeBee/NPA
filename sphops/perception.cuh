#pragma once

#include <cub/block/block_reduce.cuh>
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

// Tuning knobs. The neighbor kernels (density, forward, backward) assign one
// lane to each particle of a grid block. All particles of a grid block share one cell and
// therefore one neighbor-candidate sequence, so the lanes of a warp read the
// same shared-memory neighbor at once (broadcast, no shuffles); per-pair
// geometry is computed once and candidates outside the support are skipped.
// Each grid block runs NEIGHBOR_WARPS warp teams that take every
// NEIGHBOR_WARPS-th neighbor tile into warp-private shared buffers, then reduce
// their partial sums in fixed team order (deterministic; NEIGHBOR_WARPS = 1
// sums each particle's neighbors in one sequential sweep). FEATURES=32 splits
// channels into two groups of MAX_CHANNELS_PER_THREAD to bound registers. A
// tile holds up to NEIGHBOR_TILE candidates, halved when one warp's tile would
// exceed TILE_BYTES_PER_WARP (shared memory limits resident blocks).
constexpr int NEIGHBOR_WARPS = 4;
constexpr int MAX_CHANNELS_PER_THREAD = 16;
constexpr int NEIGHBOR_TILE = 32;
constexpr int TILE_BYTES_PER_WARP = 6144;
// Register bounds via __launch_bounds__ minimum resident warps per SM.
constexpr int FORWARD_RESIDENT_WARPS = 16;
constexpr int BACKWARD_RESIDENT_WARPS = 32;
constexpr int DENSITY_RESIDENT_WARPS = 32;
constexpr int BACKWARD_PREPARE_THREADS = 256;
static_assert(NEIGHBOR_WARPS >= 1 && NEIGHBOR_WARPS <= 8, "Use 1-8 neighbor warp teams");
static_assert(BACKWARD_PREPARE_THREADS > 0 && BACKWARD_PREPARE_THREADS <= 1024,
              "Backward preparation must use a valid CUDA block size");
static_assert(NEIGHBOR_TILE >= 4 && NEIGHBOR_TILE <= 32 && NEIGHBOR_TILE % 4 == 0,
              "A neighbor tile is loaded by one warp and keeps 16-byte alignment");

// Channels owned by one lane of a feature kernel.
template <int FEATURES>
constexpr int THREAD_CHANNELS = FEATURES > MAX_CHANNELS_PER_THREAD ? MAX_CHANNELS_PER_THREAD : FEATURES;

// Shared-memory and thread layout of a feature kernel. PAYLOAD floats per
// neighbor and channel group are staged with its position (and volume in
// forward); PARTIALS floats per lane are reduced across teams.
template <int DIM, int FEATURES, int PAYLOAD, bool VOLUME, int PARTIALS, int RESIDENT_WARPS>
struct TeamLayout {
    static constexpr int SPLIT = FEATURES > MAX_CHANNELS_PER_THREAD ? FEATURES / MAX_CHANNELS_PER_THREAD : 1;
    static constexpr int CHANNELS = FEATURES / SPLIT;
    static constexpr int WARPS = NEIGHBOR_WARPS * SPLIT;
    static constexpr int THREADS = 32 * WARPS;
    static constexpr int MIN_BLOCKS = RESIDENT_WARPS / WARPS > 0 ? RESIDENT_WARPS / WARPS : 1;
    static constexpr int POSITION_FLOATS = sizeof(ParticlePosition<DIM>) / sizeof(float);
    static constexpr int NEIGHBOR_FLOATS = PAYLOAD + POSITION_FLOATS + (VOLUME ? 1 : 0);
    static constexpr int TILE = NEIGHBOR_TILE * NEIGHBOR_FLOATS * 4 <= TILE_BYTES_PER_WARP
        ? NEIGHBOR_TILE : NEIGHBOR_TILE / 2;
    // Per-warp tile: [payload][positions][volumes]; a multiple of 4 floats keeps
    // every warp's float4 payload and positions 16-byte aligned.
    static constexpr int TILE_FLOATS = (TILE * NEIGHBOR_FLOATS + 3) / 4 * 4;
    static constexpr int REDUCERS = (NEIGHBOR_WARPS - 1) * SPLIT;
    // Reduction staging aliases the tiles; use the largest chunk that fits.
    static constexpr int CHUNK_FIT = REDUCERS > 0 ? WARPS * TILE_FLOATS / (REDUCERS * 32) : PARTIALS;
    static constexpr int CHUNK = CHUNK_FIT < PARTIALS ? (CHUNK_FIT > 0 ? CHUNK_FIT : 1) : PARTIALS;
    static constexpr int SHARED_FLOATS = WARPS * TILE_FLOATS > REDUCERS * 32 * CHUNK
        ? WARPS * TILE_FLOATS : REDUCERS * 32 * CHUNK;
    static_assert(PAYLOAD % 4 == 0, "Payloads are read as float4");
    static_assert(SHARED_FLOATS * sizeof(float) <= 48 * 1024,
                  "Static shared memory exceeds 48 KiB: reduce NEIGHBOR_WARPS or TILE_BYTES_PER_WARP");
};

// Sum per-lane partials of teams 1..NEIGHBOR_WARPS-1 into team 0, in that
// fixed order. All block threads must call this (it contains barriers).
template <int PARTIALS, typename Layout>
__device__ __forceinline__ void reduce_teams(
    float *partial, float *buffer, unsigned int team, unsigned int group, unsigned int lane)
{
    if constexpr (NEIGHBOR_WARPS > 1) {
        __syncthreads();  // Staging aliases tiles that other warps may still read.
        #pragma unroll
        for (int base = 0; base < PARTIALS; base += Layout::CHUNK) {
            if (team > 0) {
                float *slot = buffer + ((team - 1) * Layout::SPLIT + group) * Layout::CHUNK * 32 + lane;
                #pragma unroll
                for (int e = 0; e < Layout::CHUNK; ++e)
                    if (base + e < PARTIALS)
                        slot[e * 32] = partial[base + e];
            }
            __syncthreads();
            if (team == 0) {
                // Rolled: hoisting every team's loads would spill the partials.
                #pragma unroll 1
                for (int source = 1; source < NEIGHBOR_WARPS; ++source) {
                    const float *slot = buffer + ((source - 1) * Layout::SPLIT + group) * Layout::CHUNK * 32 + lane;
                    #pragma unroll
                    for (int e = 0; e < Layout::CHUNK; ++e)
                        if (base + e < PARTIALS)
                            partial[base + e] += slot[e * 32];
                }
            }
            __syncthreads();  // Staging is reused by the next chunk or pass.
        }
    }
}

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

// Kernel weights for a pair inside the support (d2 < eps^2). Callers skip
// candidates outside it, whose weights are zero. Coincident particles blur but
// have no gradient direction.
template <int DIM, BoundaryCondition BC>
__device__ __forceinline__ void support_weights(
    const GridConfig<DIM, BC, KernelStrategy::GridBased> &config,
    const ParticlePosition<DIM> &r, float d2, float h2,
    float &blur_weight, float *gradient_weight)
{
    #pragma unroll
    for (int d = 0; d < DIM; ++d)
        gradient_weight[d] = 0.0f;
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

// Forward stage 1: mass-weighted density rho_i = sum_j m_j W_ij and optional
// density gradient, with the feature kernels' lane-per-particle team layout
// (neighbor positions and masses staged per warp). Cache volume_i = m_i/rho_i;
// subsequent fused stages need volume, not another density traversal or mass read.
template <int DIM, bool DENSITY_GRAD>
struct DensityLayout {
    static constexpr int SPLIT = 1;
    static constexpr int WARPS = NEIGHBOR_WARPS;
    static constexpr int THREADS = 32 * WARPS;
    static constexpr int MIN_BLOCKS = DENSITY_RESIDENT_WARPS / WARPS > 0 ? DENSITY_RESIDENT_WARPS / WARPS : 1;
    static constexpr int TILE = NEIGHBOR_TILE;
    static constexpr int POSITION_FLOATS = sizeof(ParticlePosition<DIM>) / sizeof(float);
    // Per-warp tile: [positions][masses], a multiple of 4 floats.
    static constexpr int TILE_FLOATS = (TILE * (POSITION_FLOATS + 1) + 3) / 4 * 4;
    static constexpr int PARTIALS = 1 + (DENSITY_GRAD ? DIM : 0);
    static constexpr int CHUNK = PARTIALS;
    static constexpr int SHARED_FLOATS = WARPS * TILE_FLOATS > (WARPS - 1) * 32 * PARTIALS
        ? WARPS * TILE_FLOATS : (WARPS - 1) * 32 * PARTIALS;
};

template <int DIM, BoundaryCondition BC, bool DENSITY_GRAD>
__global__ void __launch_bounds__(DensityLayout<DIM, DENSITY_GRAD>::THREADS,
                                  DensityLayout<DIM, DENSITY_GRAD>::MIN_BLOCKS) density_forward(
    const ParticlePosition<DIM> *__restrict__ positions,
    const float *__restrict__ mass,
    float *__restrict__ volume, float *__restrict__ density_gradient,
    const uint32_t *__restrict__ bin_offsets,
    const BlockInfo *__restrict__ block_info,
    GridConfig<DIM, BC, KernelStrategy::GridBased> config)
{
    using Layout = DensityLayout<DIM, DENSITY_GRAD>;
    constexpr int TILE = Layout::TILE;
    // Snapshot-owned zero sentinels remain valid after later grid instantiations.
    BlockInfo info = block_info[blockIdx.x];
    if (info.particle_count == 0)
        return;
    __shared__ __align__(16) float shared[Layout::SHARED_FLOATS];

    constexpr int ROWS = DIM == 2 ? 3 : 9;
    constexpr int WRAP_CHUNKS = BC == BoundaryCondition::PERIODIC ? 2 : 1;
    int cell[DIM];
    config.hash2cell(info.cell_idx - info.batch_idx * config.cell_count, cell);
    unsigned int start = bin_offsets[info.cell_idx] + info.offset_in_cell;
    unsigned int lane = threadIdx.x & 31;
    unsigned int team = threadIdx.x >> 5;
    auto *tile_pos = reinterpret_cast<ParticlePosition<DIM> *>(shared + team * Layout::TILE_FLOATS);
    float *tile_mass = shared + team * Layout::TILE_FLOATS + TILE * Layout::POSITION_FLOATS;
    float h2 = config.eps * config.eps;

    for (unsigned int offset = 0; offset < info.particle_count; offset += 32) {
        bool valid = offset + lane < info.particle_count;
        unsigned int particle = start + offset + lane;
        ParticlePosition<DIM> position{};
        if (valid)
            position = positions[particle];
        float partial[Layout::PARTIALS] = {0.0f};  // rho, then density gradient
        unsigned int tile_index = 0;
        for (int row = 0; row < ROWS; ++row) {
            for (int wrap = 0; wrap < WRAP_CHUNKS; ++wrap) {
                unsigned int first, end;
                if (!neighbor_span(config, cell, info.batch_idx, row, wrap,
                                   bin_offsets, first, end))
                    continue;
                for (unsigned int remote = first; remote < end; remote += TILE) {
                    if (tile_index++ % NEIGHBOR_WARPS != team)
                        continue;
                    unsigned int count = min(unsigned(TILE), end - remote);
                    __syncwarp();
                    if (lane < count) {
                        tile_pos[lane] = positions[remote + lane];
                        tile_mass[lane] = mass[remote + lane];
                    }
                    __syncwarp();
                    if (!valid)
                        continue;
                    for (unsigned int j = 0; j < count; ++j) {
                        ParticlePosition<DIM> r = config.displacement(position, tile_pos[j]);
                        float d2 = dot(r, r);
                        if (d2 >= h2)
                            continue;
                        float neighbor_mass = tile_mass[j];
                        partial[0] += neighbor_mass * pow3(h2 - d2);
                        if constexpr (DENSITY_GRAD) {
                            if (d2 > 0.0f) {
                                float distance = sqrtf(d2);
                                float magnitude = neighbor_mass * 3.0f * pow2(config.eps - distance) / distance;
                                #pragma unroll
                                for (int d = 0; d < DIM; ++d)
                                    partial[1 + d] += magnitude * r[d];
                            }
                        }
                    }
                }
            }
        }
        reduce_teams<Layout::PARTIALS, Layout>(partial, shared, team, 0, lane);
        if (team != 0 || !valid)
            continue;
        float rho = partial[0] * config.smoothing_coef;
        volume[particle] = mass[particle] / rho;
        if constexpr (DENSITY_GRAD) {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
                density_gradient[particle * DIM + d] = partial[1 + d] * config.spiky_coef;
        }
    }
}

// Total mass of each sample, which normalizes the density-gradient feature.
// Reads masses in the original particle order and sums them in a fixed order,
// so the result does not depend on the (atomic) binning order. One block per sample.
constexpr int TOTAL_MASS_THREADS = 256;

__global__ void __launch_bounds__(TOTAL_MASS_THREADS) total_mass(
    const float *__restrict__ mass, unsigned int particle_count, float *__restrict__ total)
{
    using BlockReduce = cub::BlockReduce<float, TOTAL_MASS_THREADS>;
    __shared__ typename BlockReduce::TempStorage storage;
    const float *sample = mass + size_t(blockIdx.x) * particle_count;
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < particle_count; i += TOTAL_MASS_THREADS)
        sum += sample[i];
    sum = BlockReduce(storage).Sum(sum);
    if (threadIdx.x == 0)
        total[blockIdx.x] = sum;
}

// Forward stage 2: a lane per particle (see the tuning-knob notes). Per-pair
// arithmetic matches the original fused kernel; partial sums of the neighbor
// teams are reduced before moment correction, normalization, and packing of
// [state(C), blur(C), corrected-gradient(C*DIM), optional density-grad(DIM)].
template <int DIM, int FEATURES>
using ForwardLayout = TeamLayout<DIM, FEATURES, THREAD_CHANNELS<FEATURES>, true,
                                 THREAD_CHANNELS<FEATURES> * (1 + DIM) + DIM * (DIM + 1) / 2 + DIM,
                                 FORWARD_RESIDENT_WARPS>;

template <int DIM, int FEATURES, BoundaryCondition BC>
__global__ void __launch_bounds__(ForwardLayout<DIM, FEATURES>::THREADS,
                                  ForwardLayout<DIM, FEATURES>::MIN_BLOCKS) forward(
    const ParticlePosition<DIM> *__restrict__ positions,
    const float *__restrict__ state, const float *__restrict__ volume,
    const float *__restrict__ density_gradient, const float *__restrict__ total_mass,
    const float *__restrict__ eps0, float *__restrict__ perception, float *__restrict__ gradient_values,
    float *__restrict__ gradient_sum,
    const uint32_t *__restrict__ bin_offsets,
    const BlockInfo *__restrict__ block_info,
    GridConfig<DIM, BC, KernelStrategy::GridBased> config,
    bool perceive_density_grad, bool normalize_state_grad,
    bool normalize_density_grad, bool save_backward)
{
    static_assert(FEATURES == 4 || FEATURES == 8 || FEATURES == 16 || FEATURES == 32,
                  "Native feature counts are vectorizable powers of two");
    using Layout = ForwardLayout<DIM, FEATURES>;
    constexpr int CHANNELS = Layout::CHANNELS;
    constexpr int TILE = Layout::TILE;
    // Per-lane partial sums: blur, raw gradient, upper moment matrix, and the
    // volume-weighted kernel-gradient sum used by backward.
    constexpr int BLUR = 0, GRADIENT = CHANNELS, MOMENT = CHANNELS * (1 + DIM);
    constexpr int WEIGHTED_SUM = MOMENT + DIM * (DIM + 1) / 2;
    constexpr int PARTIALS = WEIGHTED_SUM + DIM;
    static_assert(CHANNELS == THREAD_CHANNELS<FEATURES>, "Layout and kernel channel groups differ");
    BlockInfo info = block_info[blockIdx.x];
    if (info.particle_count == 0)
        return;
    __shared__ __align__(16) float shared[Layout::SHARED_FLOATS];

    constexpr int ROWS = DIM == 2 ? 3 : 9;
    constexpr int WRAP_CHUNKS = BC == BoundaryCondition::PERIODIC ? 2 : 1;
    int cell[DIM];
    config.hash2cell(info.cell_idx - info.batch_idx * config.cell_count, cell);
    unsigned int start = bin_offsets[info.cell_idx] + info.offset_in_cell;
    unsigned int lane = threadIdx.x & 31;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int group = warp % Layout::SPLIT;
    unsigned int team = warp / Layout::SPLIT;
    unsigned int first_channel = group * CHANNELS;
    float *tile_state = shared + warp * Layout::TILE_FLOATS;
    auto *tile_pos = reinterpret_cast<ParticlePosition<DIM> *>(tile_state + TILE * CHANNELS);
    float *tile_volume = tile_state + TILE * (CHANNELS + Layout::POSITION_FLOATS);
    unsigned int output_dim = FEATURES * (2 + DIM) + (perceive_density_grad ? DIM : 0);
    float scale = config.eps / eps0[0];
    float h2 = config.eps * config.eps;

    for (unsigned int offset = 0; offset < info.particle_count; offset += 32) {
        // Lanes without a particle still load tiles and reach every barrier,
        // but skip pair work and output.
        bool valid = offset + lane < info.particle_count;
        unsigned int particle = start + offset + lane;
        ParticlePosition<DIM> position{};
        float local_state[CHANNELS] = {0.0f};
        float partial[PARTIALS] = {0.0f};
        if (valid) {
            position = positions[particle];
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c)
                local_state[c] = state[particle * FEATURES + first_channel + c];
        }

        unsigned int tile_index = 0;
        for (int row = 0; row < ROWS; ++row) {
            for (int wrap = 0; wrap < WRAP_CHUNKS; ++wrap) {
                unsigned int first, end;
                if (!neighbor_span(config, cell, info.batch_idx, row, wrap,
                                   bin_offsets, first, end))
                    continue;
                for (unsigned int remote = first; remote < end; remote += TILE) {
                    if (tile_index++ % NEIGHBOR_WARPS != team)
                        continue;
                    unsigned int count = min(unsigned(TILE), end - remote);
                    __syncwarp();  // Finish reading the previous tile before refilling it.
                    if (lane < count) {
                        tile_pos[lane] = positions[remote + lane];
                        tile_volume[lane] = volume[remote + lane];
                    }
                    #pragma unroll 4
                    for (unsigned int k = lane; k < count * CHANNELS; k += 32)
                        tile_state[k] = state[(remote + k / CHANNELS) * FEATURES + first_channel + k % CHANNELS];
                    __syncwarp();  // Tile loads must be visible to the whole warp.
                    if (!valid)
                        continue;
                    for (unsigned int j = 0; j < count; ++j) {
                        ParticlePosition<DIM> r = config.displacement(position, tile_pos[j]);
                        float d2 = dot(r, r);
                        if (d2 >= h2)
                            continue;
                        float weight, grad_weight[DIM];
                        support_weights(config, r, d2, h2, weight, grad_weight);
                        float v = tile_volume[j];
                        int moment = MOMENT;
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d) {
                            if (save_backward)
                                partial[WEIGHTED_SUM + d] += v * grad_weight[d];
                            #pragma unroll
                            for (int e = d; e < DIM; ++e)
                                partial[moment++] += v * grad_weight[d] * r[e];
                        }
                        const float4 *remote_state = reinterpret_cast<const float4 *>(&tile_state[j * CHANNELS]);
                        #pragma unroll
                        for (int q = 0; q < CHANNELS / 4; ++q) {
                            float4 packed = remote_state[q];
                            float values[4] = {packed.x, packed.y, packed.z, packed.w};
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                int c = 4 * q + k;
                                partial[BLUR + c] += values[k] * (weight * v);
                                float difference = (values[k] - local_state[c]) * v;
                                #pragma unroll
                                for (int d = 0; d < DIM; ++d)
                                    partial[GRADIENT + c * DIM + d] += difference * grad_weight[d];
                            }
                        }
                    }
                }
            }
        }
        reduce_teams<PARTIALS, Layout>(partial, shared, team, group, lane);
        if (team != 0 || !valid)
            continue;

        float matrix[DIM * DIM];
        int moment = MOMENT;
        #pragma unroll
        for (int d = 0; d < DIM; ++d)
            #pragma unroll
            for (int e = d; e < DIM; ++e)
                matrix[d * DIM + e] = matrix[e * DIM + d] = partial[moment++];
        safe_inverse<DIM>(matrix);

        unsigned int base = particle * output_dim;
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            unsigned int channel = first_channel + c;
            // Moment correction changes the forward value only: gradient_hybrid
            // uses raw + (raw @ inverse - raw).detach(). Its adjoint is handled below.
            float corrected[DIM] = {0.0f};
            #pragma unroll
            for (int d = 0; d < DIM; ++d) {
                #pragma unroll
                for (int e = 0; e < DIM; ++e)
                    corrected[d] += partial[GRADIENT + c * DIM + e] * matrix[e * DIM + d];
                corrected[d] *= scale;
                if (save_backward && normalize_state_grad)
                    gradient_values[(particle * FEATURES + channel) * DIM + d] = corrected[d];
            }
            if (normalize_state_grad)
                log_normalize<DIM>(corrected);
            perception[base + channel] = local_state[c];
            perception[base + FEATURES + channel] = partial[BLUR + c];
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
                perception[base + 2 * FEATURES + channel * DIM + d] = corrected[d];
        }
        if (group == 0) {
            if (save_backward) {
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                    gradient_sum[particle * DIM + d] = partial[WEIGHTED_SUM + d];
            }
            if (perceive_density_grad) {
                float density_scale = 1.0f;
                #pragma unroll
                for (int d = 0; d < DIM + 1; ++d)
                    density_scale *= scale;
                // Divide by the sample's total mass (null for unit masses: N).
                density_scale /= total_mass ? total_mass[info.batch_idx] : float(config.particle_count);
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
// one traversal, with the forward kernel's lane-per-particle team layout. Team
// 0 owns the final grad_state entries: no global atomics. volume_i multiplies
// incoming neighbor contributions; the local/self term was already written by
// backward_prepare. No position, density, or mass adjoints.
template <int DIM, int FEATURES>
using BackwardLayout = TeamLayout<DIM, FEATURES, (DIM + 1) * THREAD_CHANNELS<FEATURES>, false,
                                  THREAD_CHANNELS<FEATURES>, BACKWARD_RESIDENT_WARPS>;

template <int DIM, int FEATURES, BoundaryCondition BC>
__global__ void __launch_bounds__(BackwardLayout<DIM, FEATURES>::THREADS,
                                  BackwardLayout<DIM, FEATURES>::MIN_BLOCKS) backward(
    const ParticlePosition<DIM> *__restrict__ positions,
    const float *__restrict__ volume, const float *__restrict__ incoming,
    float *__restrict__ grad_state, const uint32_t *__restrict__ bin_offsets,
    const BlockInfo *__restrict__ block_info,
    GridConfig<DIM, BC, KernelStrategy::GridBased> config)
{
    static_assert(FEATURES == 4 || FEATURES == 8 || FEATURES == 16 || FEATURES == 32,
                  "Native feature counts are vectorizable powers of two");
    using Layout = BackwardLayout<DIM, FEATURES>;
    constexpr int CHANNELS = Layout::CHANNELS;
    constexpr int TILE = Layout::TILE;
    constexpr int STRIDE = (DIM + 1) * FEATURES;  // Global incoming row per particle.
    constexpr int PAYLOAD = (DIM + 1) * CHANNELS;  // This channel group's share.
    BlockInfo info = block_info[blockIdx.x];
    if (info.particle_count == 0)
        return;
    __shared__ __align__(16) float shared[Layout::SHARED_FLOATS];

    constexpr int ROWS = DIM == 2 ? 3 : 9;
    constexpr int WRAP_CHUNKS = BC == BoundaryCondition::PERIODIC ? 2 : 1;
    int cell[DIM];
    config.hash2cell(info.cell_idx - info.batch_idx * config.cell_count, cell);
    unsigned int start = bin_offsets[info.cell_idx] + info.offset_in_cell;
    unsigned int lane = threadIdx.x & 31;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int group = warp % Layout::SPLIT;
    unsigned int team = warp / Layout::SPLIT;
    unsigned int first_channel = group * CHANNELS;
    float *tile_incoming = shared + warp * Layout::TILE_FLOATS;
    auto *tile_pos = reinterpret_cast<ParticlePosition<DIM> *>(tile_incoming + TILE * PAYLOAD);
    float h2 = config.eps * config.eps;

    for (unsigned int offset = 0; offset < info.particle_count; offset += 32) {
        bool valid = offset + lane < info.particle_count;
        unsigned int particle = start + offset + lane;
        ParticlePosition<DIM> position{};
        if (valid)
            position = positions[particle];
        float result[CHANNELS] = {0.0f};
        unsigned int tile_index = 0;
        for (int row = 0; row < ROWS; ++row) {
            for (int wrap = 0; wrap < WRAP_CHUNKS; ++wrap) {
                unsigned int first, end;
                if (!neighbor_span(config, cell, info.batch_idx, row, wrap,
                                   bin_offsets, first, end))
                    continue;
                for (unsigned int remote = first; remote < end; remote += TILE) {
                    if (tile_index++ % NEIGHBOR_WARPS != team)
                        continue;
                    unsigned int count = min(unsigned(TILE), end - remote);
                    __syncwarp();
                    if (lane < count)
                        tile_pos[lane] = positions[remote + lane];
                    // Tile layout [neighbor][component][channel of this group].
                    #pragma unroll 4
                    for (unsigned int k = lane; k < count * PAYLOAD; k += 32)
                        tile_incoming[k] = incoming[(remote + k / PAYLOAD) * STRIDE
                                                    + k % PAYLOAD / CHANNELS * FEATURES
                                                    + first_channel + k % CHANNELS];
                    __syncwarp();
                    if (!valid)
                        continue;
                    for (unsigned int j = 0; j < count; ++j) {
                        ParticlePosition<DIM> r = config.displacement(position, tile_pos[j]);
                        float d2 = dot(r, r);
                        if (d2 >= h2)
                            continue;
                        float weight, grad_weight[DIM];
                        support_weights(config, r, d2, h2, weight, grad_weight);
                        const float *cotangent = &tile_incoming[j * PAYLOAD];
                        #pragma unroll
                        for (int q = 0; q < CHANNELS / 4; ++q) {
                            float4 packed[DIM + 1];
                            #pragma unroll
                            for (int k = 0; k <= DIM; ++k)
                                packed[k] = reinterpret_cast<const float4 *>(cotangent + k * CHANNELS)[q];
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                int c = 4 * q + k;
                                result[c] += weight * reinterpret_cast<const float *>(&packed[0])[k];
                                #pragma unroll
                                for (int d = 0; d < DIM; ++d)
                                    // The gradient at j uses displacement j -> i, the negative
                                    // of the forward displacement i -> j computed above.
                                    result[c] -= grad_weight[d] * reinterpret_cast<const float *>(&packed[d + 1])[k];
                            }
                        }
                    }
                }
            }
        }
        reduce_teams<CHANNELS, Layout>(result, shared, team, group, lane);
        if (team == 0 && valid) {
            float particle_volume = volume[particle];
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c)
                grad_state[particle * FEATURES + first_channel + c] += particle_volume * result[c];
        }
    }
}

} // namespace fused_perception
