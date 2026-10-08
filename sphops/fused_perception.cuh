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
// Tile buffers per warp: 2 lets a warp copy its next tile (cp.async) while it
// processes the current one. The tile shrinks so that all buffers fit
// TILE_BYTES_PER_WARP.
constexpr int FEATURE_TILE_BUFFERS = 2;
constexpr int BACKWARD_TILE_BUFFERS = 2;
constexpr int DENSITY_TILE_BUFFERS = 2;
// Team-reduction staging may grow beyond the tile buffers up to this size, so
// that the partials of a lane are exchanged in as few chunks (barrier pairs)
// as possible without reducing the number of resident blocks.
constexpr int REDUCE_STAGING_BYTES = 21504;
// Register bounds via __launch_bounds__ minimum resident warps per SM.
constexpr int FORWARD_RESIDENT_WARPS = 16;
constexpr int BACKWARD_RESIDENT_WARPS = 32;
constexpr int DENSITY_RESIDENT_WARPS = 32;
constexpr int BACKWARD_PREPARE_THREADS = 256;
// gradient_hybrid's determinant threshold for the moment-matrix inverse.
constexpr float MOMENT_INVERSE_TOLERANCE = 1e-3f;
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
// forward); PARTIALS floats per lane are reduced across teams; OUTPUT_FLOATS
// is the coalesced-output staging (all three alias the same buffer).
template <int DIM, int FEATURES, int PAYLOAD, bool VOLUME, int PARTIALS, int RESIDENT_WARPS, int OUTPUT_FLOATS,
          int TILE_BUFFERS>
struct TeamLayout {
    static constexpr int BUFFERS = TILE_BUFFERS;
    static constexpr int SPLIT = FEATURES > MAX_CHANNELS_PER_THREAD ? FEATURES / MAX_CHANNELS_PER_THREAD : 1;
    static constexpr int CHANNELS = FEATURES / SPLIT;
    static constexpr int WARPS = NEIGHBOR_WARPS * SPLIT;
    static constexpr int THREADS = 32 * WARPS;
    static constexpr int MIN_BLOCKS = RESIDENT_WARPS / WARPS > 0 ? RESIDENT_WARPS / WARPS : 1;
    static constexpr int POSITION_FLOATS = sizeof(ParticlePosition<DIM>) / sizeof(float);
    static constexpr int NEIGHBOR_FLOATS = PAYLOAD + POSITION_FLOATS + (VOLUME ? 1 : 0);
    static constexpr int TILE = BUFFERS * NEIGHBOR_TILE * NEIGHBOR_FLOATS * 4 <= TILE_BYTES_PER_WARP ? NEIGHBOR_TILE
        : BUFFERS * (NEIGHBOR_TILE / 2) * NEIGHBOR_FLOATS * 4 <= TILE_BYTES_PER_WARP ? NEIGHBOR_TILE / 2
        : NEIGHBOR_TILE / 4;
    // Per-warp tile: [payload][positions][volumes]; a multiple of 4 floats keeps
    // every warp's float4 payload 16-byte aligned.
    static constexpr int TILE_FLOATS = (TILE * NEIGHBOR_FLOATS + 3) / 4 * 4;
    static constexpr int WARP_FLOATS = BUFFERS * TILE_FLOATS;
    static constexpr int REDUCERS = (NEIGHBOR_WARPS - 1) * SPLIT;
    // Reduction staging aliases the tiles (and may extend them up to
    // REDUCE_STAGING_BYTES); use the largest chunk that fits.
    static constexpr int STAGING_FLOATS = WARPS * WARP_FLOATS > REDUCE_STAGING_BYTES / 4
        ? WARPS * WARP_FLOATS : REDUCE_STAGING_BYTES / 4;
    static constexpr int CHUNK_FIT = REDUCERS > 0 ? STAGING_FLOATS / (REDUCERS * 32) : PARTIALS;
    static constexpr int CHUNK = CHUNK_FIT < PARTIALS ? (CHUNK_FIT > 0 ? CHUNK_FIT : 1) : PARTIALS;
    static constexpr int TEAM_FLOATS = WARPS * WARP_FLOATS > REDUCERS * 32 * CHUNK
        ? WARPS * WARP_FLOATS : REDUCERS * 32 * CHUNK;
    // All-gather of partials (every warp receives every team's sums).
    static constexpr int GATHER_FIT = STAGING_FLOATS / (WARPS * 32);
    static constexpr int GATHER_CHUNK = GATHER_FIT < PARTIALS ? (GATHER_FIT > 0 ? GATHER_FIT : 1) : PARTIALS;
    static constexpr int GATHER_FLOATS = WARPS * 32 * GATHER_CHUNK;
    static constexpr int EXCHANGE_FLOATS = TEAM_FLOATS > GATHER_FLOATS ? TEAM_FLOATS : GATHER_FLOATS;
    static constexpr int SHARED_FLOATS = EXCHANGE_FLOATS > OUTPUT_FLOATS ? EXCHANGE_FLOATS : OUTPUT_FLOATS;
    static_assert(PAYLOAD % 4 == 0, "Payloads are read as float4");
    static_assert(TILE >= 4, "A tile buffer holds at least four candidates");
    static_assert(SHARED_FLOATS * sizeof(float) <= 48 * 1024,
                  "Static shared memory exceeds 48 KiB: reduce NEIGHBOR_WARPS or TILE_BYTES_PER_WARP");
};

// log2 of the lanes per particle in one pass over a grid block: a sparse block
// (few particles left) gives each particle several lanes, which split its
// candidates (lane r of the particle takes candidates j = r, r + share, ...).
// Block-uniform.
__device__ __forceinline__ unsigned int lane_share_shift(unsigned int remaining)
{
    return remaining <= 4 ? 3u : remaining <= 8 ? 2u : remaining <= 16 ? 1u : 0u;
}

// Largest useful lanes-per-particle shift for a block's candidate density
// (average candidates per neighbor row span, from the block record): the
// butterfly costs PARTIALS shuffles per step, so sharing pays only when the
// lanes of a particle have enough candidates per tile to split.
__device__ __forceinline__ unsigned int lane_share_cap(unsigned int per_row)
{
    return per_row >= 16 ? 3u : per_row >= 8 ? 2u : per_row >= 4 ? 1u : 0u;
}

// Sum the partials of the `share` lanes of each particle with an xor butterfly,
// so every lane of the group ends with the same total (a + b = b + a exactly,
// and the tree is the same in all lanes). `share` is warp-uniform; all lanes call this.
template <int N>
__device__ __forceinline__ void reduce_lanes(float *values, unsigned int share)
{
    for (unsigned int step = 1; step < share; step <<= 1) {
        #pragma unroll
        for (int e = 0; e < N; ++e)
            values[e] += __shfl_xor_sync(0xffffffffu, values[e], step);
    }
}

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

// Like reduce_teams, but every warp ends with the same total, summed in the
// same fixed order (team 0, 1, 2, ...), so all teams can share the epilogue.
template <int PARTIALS, typename Layout>
__device__ __forceinline__ void gather_teams(
    float *partial, float *buffer, unsigned int team, unsigned int group, unsigned int lane)
{
    if constexpr (NEIGHBOR_WARPS > 1) {
        __syncthreads();  // Staging aliases tiles that other warps may still read.
        #pragma unroll
        for (int base = 0; base < PARTIALS; base += Layout::GATHER_CHUNK) {
            float *slot = buffer + (team * Layout::SPLIT + group) * Layout::GATHER_CHUNK * 32 + lane;
            #pragma unroll
            for (int e = 0; e < Layout::GATHER_CHUNK; ++e)
                if (base + e < PARTIALS)
                    slot[e * 32] = partial[base + e];
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < Layout::GATHER_CHUNK; ++e) {
                if (base + e < PARTIALS) {
                    float sum = buffer[group * Layout::GATHER_CHUNK * 32 + lane + e * 32];
                    #pragma unroll 1
                    for (int source = 1; source < NEIGHBOR_WARPS; ++source)
                        sum += buffer[(source * Layout::SPLIT + group) * Layout::GATHER_CHUNK * 32 + lane + e * 32];
                    partial[base + e] = sum;
                }
            }
            __syncthreads();  // Staging is reused by the next chunk or pass.
        }
    }
}

// Asynchronous global-to-shared copies (cp.async, sm_80+; plain copies on
// older targets). Each thread's copies form groups; wait_tiles<N> blocks until
// at most N of this thread's groups are still in flight.
__device__ __forceinline__ void copy_async16(void *shared_dst, const void *global_src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned int dst = static_cast<unsigned int>(__cvta_generic_to_shared(shared_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(global_src));
#else
    *reinterpret_cast<float4 *>(shared_dst) = *reinterpret_cast<const float4 *>(global_src);
#endif
}

__device__ __forceinline__ void copy_async8(void *shared_dst, const void *global_src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned int dst = static_cast<unsigned int>(__cvta_generic_to_shared(shared_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(global_src));
#else
    *reinterpret_cast<float2 *>(shared_dst) = *reinterpret_cast<const float2 *>(global_src);
#endif
}

__device__ __forceinline__ void copy_async4(void *shared_dst, const void *global_src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned int dst = static_cast<unsigned int>(__cvta_generic_to_shared(shared_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(global_src));
#else
    *reinterpret_cast<float *>(shared_dst) = *reinterpret_cast<const float *>(global_src);
#endif
}

template <int DIM>
__device__ __forceinline__ void copy_async_position(ParticlePosition<DIM> *shared_dst, const ParticlePosition<DIM> *global_src)
{
    if constexpr (DIM == 2)
        copy_async8(shared_dst, global_src);
    else {
        copy_async4(reinterpret_cast<float *>(shared_dst), reinterpret_cast<const float *>(global_src));
        copy_async4(reinterpret_cast<float *>(shared_dst) + 1, reinterpret_cast<const float *>(global_src) + 1);
        copy_async4(reinterpret_cast<float *>(shared_dst) + 2, reinterpret_cast<const float *>(global_src) + 2);
    }
}

__device__ __forceinline__ void commit_tile()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

template <int PENDING>
__device__ __forceinline__ void wait_tiles()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(PENDING));
#endif
}

// Walks the tiles of one team: the spans' candidates are cut into tiles of at
// most TILE that never cross a span, and tile k goes to team k % NEIGHBOR_WARPS.
template <int TILE, int DIM, BoundaryCondition BC>
struct TileCursor {
    static constexpr int WRAPS = GridConfig<DIM, BC>::WRAP_CHUNKS;
    static constexpr int SPANS = GridConfig<DIM, BC>::NEIGHBOR_ROWS * WRAPS;
    const GridConfig<DIM, BC> &config;
    const int *cell;
    unsigned int batch;
    const uint32_t *bin_offsets;
    unsigned int team;
    int span = -1;
    unsigned int next = 0, end = 0, index = 0;
    unsigned int remote = 0, count = 0;
    bool ok = false;

    __device__ __forceinline__ TileCursor(const GridConfig<DIM, BC> &config, const int *cell, unsigned int batch,
                                          const uint32_t *bin_offsets, unsigned int team)
        : config(config), cell(cell), batch(batch), bin_offsets(bin_offsets), team(team)
    {
        advance();
    }

    __device__ __forceinline__ void advance()
    {
        while (true) {
            if (next < end) {
                unsigned int size = min(unsigned(TILE), end - next);
                unsigned int tile = index++;
                unsigned int at = next;
                next += size;
                if (tile % NEIGHBOR_WARPS == team) {
                    remote = at;
                    count = size;
                    ok = true;
                    return;
                }
                continue;
            }
            if (++span >= SPANS) {
                ok = false;
                return;
            }
            if (!config.neighbor_span(cell, batch, span / WRAPS, span % WRAPS, bin_offsets, next, end))
                next = end = 0;
        }
    }
};

// Visits the neighbor candidates of this grid block's particles. Each warp
// stages its tiles in its own BUFFERS shared buffers of TILE_FLOATS each
// (tiles + b * TILE_FLOATS) with load(remote, count, tile): asynchronous copies
// that commit as one group, so with two buffers the next tile streams in while
// the current one is processed. Every valid lane then calls pair(j, r, d2, tile)
// for each candidate j of the tile inside the kernel support, where r is the
// displacement to j and d2 = |r|^2, and tile + POSITION_OFFSET holds the tile's
// positions. All lanes of the warp must call this (the barriers are per warp).
// `sub` and `shift`: this lane handles candidates j = sub, sub + 2^shift, ...
// of every tile (2^shift lanes per particle, see lane_share_shift).
template <int TILE, int TILE_FLOATS, int BUFFERS, int POSITION_OFFSET, int DIM, BoundaryCondition BC,
          typename Load, typename Pair>
__device__ __forceinline__ void for_each_neighbor(
    const GridConfig<DIM, BC> &config, const int *cell, unsigned int batch, const uint32_t *bin_offsets,
    unsigned int team, bool valid, unsigned int sub, unsigned int shift, const ParticlePosition<DIM> &position,
    float *tiles, float h2, Load load, Pair pair)
{
    TileCursor<TILE, DIM, BC> cursor(config, cell, batch, bin_offsets, team);
    if (!cursor.ok)
        return;
    load(cursor.remote, cursor.count, tiles);
    commit_tile();
    unsigned int current = 0;
    while (true) {
        unsigned int count = cursor.count;
        cursor.advance();
        float *tile = tiles + current * TILE_FLOATS;
        if constexpr (BUFFERS > 1) {
            if (cursor.ok) {
                load(cursor.remote, cursor.count, tiles + (current ^ 1u) * TILE_FLOATS);
                commit_tile();
                wait_tiles<1>();  // This thread's copies of the current tile are done.
            } else
                wait_tiles<0>();
        } else
            wait_tiles<0>();
        __syncwarp();  // Every lane's copies of the current tile are visible.
        if (valid) {
            const auto *tile_pos = reinterpret_cast<const ParticlePosition<DIM> *>(tile + POSITION_OFFSET);
            auto candidate = [&](unsigned int j) {
                ParticlePosition<DIM> r = config.displacement(position, tile_pos[j]);
                float d2 = dot(r, r);
                if (d2 < h2)
                    pair(j, r, d2, tile);
            };
            // One lane per particle keeps the plain stride-1 loop (its
            // addresses fold into immediates); shared particles interleave
            // their lanes over the tile, which keeps the lanes' shared-memory
            // reads in distinct banks.
            if (shift == 0) {
                for (unsigned int j = 0; j < count; ++j)
                    candidate(j);
            } else {
                for (unsigned int j = sub; j < count; j += 1u << shift)
                    candidate(j);
            }
        }
        __syncwarp();  // Finish reading this buffer before it is refilled.
        if (!cursor.ok)
            break;
        if constexpr (BUFFERS > 1) {
            current ^= 1u;
        } else {
            load(cursor.remote, cursor.count, tiles);
            commit_tile();
        }
    }
}

// Row staging for coalesced output. A grid block's particles are consecutive in
// binned order, so their output rows form one contiguous global range. Lanes
// write their own row into shared memory (an odd row stride keeps these
// lane-strided writes free of bank conflicts), then all block threads copy the
// range with consecutive threads on consecutive floats. Callers synchronize the
// block before staging (the tiles alias this memory) and after the copy.
__device__ __forceinline__ unsigned int staging_stride(unsigned int width)
{
    return width | 1u;
}

template <int THREADS>
__device__ __forceinline__ void store_rows(
    float *__restrict__ destination, const float *__restrict__ stage,
    unsigned int rows, unsigned int width)
{
    unsigned int stride = staging_stride(width);
    for (unsigned int k = threadIdx.x; k < rows * width; k += THREADS) {
        unsigned int row = k / width;
        destination[k] = stage[row * stride + k - row * width];
    }
}

// Kernel weights for a pair inside the support (d2 < eps^2). Callers skip
// candidates outside it, whose weights are zero. Coincident particles blur but
// have no gradient direction.
template <int DIM, BoundaryCondition BC>
__device__ __forceinline__ void support_weights(
    const GridConfig<DIM, BC> &config,
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
    static constexpr int BUFFERS = DENSITY_TILE_BUFFERS;
    static constexpr int POSITION_FLOATS = sizeof(ParticlePosition<DIM>) / sizeof(float);
    // Per-warp tile: [positions][masses], a multiple of 4 floats.
    static constexpr int TILE_FLOATS = (TILE * (POSITION_FLOATS + 1) + 3) / 4 * 4;
    static constexpr int WARP_FLOATS = BUFFERS * TILE_FLOATS;
    static constexpr int PARTIALS = 1 + (DENSITY_GRAD ? DIM : 0);
    static constexpr int CHUNK = PARTIALS;
    static constexpr int SHARED_FLOATS = WARPS * WARP_FLOATS > (WARPS - 1) * 32 * PARTIALS
        ? WARPS * WARP_FLOATS : (WARPS - 1) * 32 * PARTIALS;
};

template <int DIM, BoundaryCondition BC, bool DENSITY_GRAD>
__global__ void __launch_bounds__(DensityLayout<DIM, DENSITY_GRAD>::THREADS,
                                  DensityLayout<DIM, DENSITY_GRAD>::MIN_BLOCKS) density_forward(
    const ParticlePosition<DIM> *__restrict__ positions,
    const float *__restrict__ mass,
    float *__restrict__ volume, float *__restrict__ density_gradient,
    const uint32_t *__restrict__ bin_offsets,
    const BlockInfo *__restrict__ block_info,
    GridConfig<DIM, BC> config)
{
    using Layout = DensityLayout<DIM, DENSITY_GRAD>;
    constexpr int TILE = Layout::TILE;
    // Snapshot-owned zero sentinels remain valid after later grid instantiations.
    // Persistent loop over grid blocks: the active records are a contiguous
    // prefix of block_info, so the first zero-count sentinel ends the work of
    // this CUDA block without scheduling the empty tail (large in sparse 3D).
    unsigned int max_blocks = config.max_blocks_per_batch * config.batch_size;
    for (unsigned int block = blockIdx.x; block < max_blocks; block += gridDim.x) {
    BlockInfo info = block_info[block];
    if (info.particle_count == 0)
        break;
    __shared__ __align__(16) float shared[Layout::SHARED_FLOATS];

    int cell[DIM];
    config.hash2cell(info.cell_idx - info.batch_idx * config.cell_count, cell);
    unsigned int share_cap = lane_share_cap(info.candidates_per_row);
    unsigned int start = bin_offsets[info.cell_idx] + info.offset_in_cell;
    unsigned int lane = threadIdx.x & 31;
    unsigned int team = threadIdx.x >> 5;
    float *tiles = shared + team * Layout::WARP_FLOATS;  // [positions][masses] per buffer
    constexpr int MASS_OFFSET = TILE * Layout::POSITION_FLOATS;
    float h2 = config.eps * config.eps;

    for (unsigned int offset = 0; offset < info.particle_count;) {
        unsigned int shift = min(lane_share_shift(info.particle_count - offset), share_cap);
        unsigned int share = 1u << shift, slot = lane >> shift, sub = lane & (share - 1);
        bool valid = offset + slot < info.particle_count;
        unsigned int particle = start + offset + slot;
        ParticlePosition<DIM> position{};
        if (valid)
            position = positions[particle];
        float partial[Layout::PARTIALS] = {0.0f};  // rho, then density gradient
        for_each_neighbor<TILE, Layout::TILE_FLOATS, Layout::BUFFERS, 0>(
            config, cell, info.batch_idx, bin_offsets, team, valid, sub, shift, position, tiles, h2,
            [&](unsigned int remote, unsigned int count, float *tile) {
                if (lane < count) {
                    copy_async_position<DIM>(reinterpret_cast<ParticlePosition<DIM> *>(tile) + lane, positions + remote + lane);
                    copy_async4(tile + MASS_OFFSET + lane, mass + remote + lane);
                }
            },
            [&](unsigned int j, const ParticlePosition<DIM> &r, float d2, const float *tile) {
                float neighbor_mass = tile[MASS_OFFSET + j];
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
            });
        reduce_lanes<Layout::PARTIALS>(partial, share);
        reduce_teams<Layout::PARTIALS, Layout>(partial, shared, team, 0, lane);
        offset += 32u >> shift;
        if (team != 0 || !valid || sub != 0)
            continue;
        float rho = partial[0] * config.smoothing_coef;
        volume[particle] = mass[particle] / rho;
        if constexpr (DENSITY_GRAD) {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
                density_gradient[particle * DIM + d] = partial[1 + d] * config.spiky_coef;
        }
    }
    __syncthreads();  // The next grid block reuses the tiles and staging.
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
// 32 staged rows of the widest output (the perception row with density
// gradient; gradient_values rows, FEATURES * DIM floats, are narrower).
template <int DIM, int FEATURES>
using ForwardLayout = TeamLayout<DIM, FEATURES, THREAD_CHANNELS<FEATURES>, true,
                                 THREAD_CHANNELS<FEATURES> * (1 + DIM) + DIM * (DIM + 1) / 2 + DIM,
                                 FORWARD_RESIDENT_WARPS,
                                 32 * ((FEATURES * (2 + DIM) + DIM) | 1) + 32 * ((FEATURES * DIM) | 1),
                                 FEATURE_TILE_BUFFERS>;

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
    GridConfig<DIM, BC> config,
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
    // Persistent loop over grid blocks: the active records are a contiguous
    // prefix of block_info, so the first zero-count sentinel ends the work of
    // this CUDA block without scheduling the empty tail (large in sparse 3D).
    unsigned int max_blocks = config.max_blocks_per_batch * config.batch_size;
    for (unsigned int block = blockIdx.x; block < max_blocks; block += gridDim.x) {
    BlockInfo info = block_info[block];
    if (info.particle_count == 0)
        break;
    __shared__ __align__(16) float shared[Layout::SHARED_FLOATS];

    int cell[DIM];
    config.hash2cell(info.cell_idx - info.batch_idx * config.cell_count, cell);
    unsigned int share_cap = lane_share_cap(info.candidates_per_row);
    unsigned int start = bin_offsets[info.cell_idx] + info.offset_in_cell;
    unsigned int lane = threadIdx.x & 31;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int group = warp % Layout::SPLIT;
    unsigned int team = warp / Layout::SPLIT;
    unsigned int first_channel = group * CHANNELS;
    // Per buffer: [states][positions][volumes].
    float *tiles = shared + warp * Layout::WARP_FLOATS;
    constexpr int POSITION_OFFSET = TILE * CHANNELS;
    constexpr int VOLUME_OFFSET = TILE * (CHANNELS + Layout::POSITION_FLOATS);
    unsigned int output_dim = FEATURES * (2 + DIM) + (perceive_density_grad ? DIM : 0);
    float scale = config.eps / eps0[0];
    float h2 = config.eps * config.eps;

    for (unsigned int offset = 0; offset < info.particle_count;) {
        // Lanes without a particle still load tiles and reach every barrier,
        // but skip pair work and output. Sparse blocks give each particle
        // several lanes, which split its candidates.
        unsigned int shift = min(lane_share_shift(info.particle_count - offset), share_cap);
        unsigned int share = 1u << shift, slot = lane >> shift, sub = lane & (share - 1);
        bool valid = offset + slot < info.particle_count;
        unsigned int particle = start + offset + slot;
        ParticlePosition<DIM> position{};
        float local_state[CHANNELS] = {0.0f};
        float partial[PARTIALS] = {0.0f};
        if (valid) {
            position = positions[particle];
            // Rows are 16-byte aligned (FEATURES and channel groups are multiples of 4).
            const float4 *row = reinterpret_cast<const float4 *>(state + particle * FEATURES + first_channel);
            #pragma unroll
            for (int q = 0; q < CHANNELS / 4; ++q) {
                float4 packed = row[q];
                local_state[4 * q] = packed.x;
                local_state[4 * q + 1] = packed.y;
                local_state[4 * q + 2] = packed.z;
                local_state[4 * q + 3] = packed.w;
            }
        }

        for_each_neighbor<TILE, Layout::TILE_FLOATS, Layout::BUFFERS, POSITION_OFFSET>(
            config, cell, info.batch_idx, bin_offsets, team, valid, sub, shift, position, tiles, h2,
            [&](unsigned int remote, unsigned int count, float *tile) {
                if (lane < count) {
                    copy_async_position<DIM>(reinterpret_cast<ParticlePosition<DIM> *>(tile + POSITION_OFFSET) + lane,
                                             positions + remote + lane);
                    copy_async4(tile + VOLUME_OFFSET + lane, volume + remote + lane);
                }
                constexpr unsigned int QUADS = CHANNELS / 4;  // float4 chunks per state row
                #pragma unroll 4
                for (unsigned int k = lane; k < count * QUADS; k += 32)
                    copy_async16(tile + (k / QUADS) * CHANNELS + (k % QUADS) * 4,
                                 state + (remote + k / QUADS) * FEATURES + first_channel + (k % QUADS) * 4);
            },
            [&](unsigned int j, const ParticlePosition<DIM> &r, float d2, const float *tile) {
                float weight, grad_weight[DIM];
                support_weights(config, r, d2, h2, weight, grad_weight);
                float v = tile[VOLUME_OFFSET + j];
                // Volume-weighted kernel weights once per pair; every channel
                // then costs one fma for the blur and DIM for the gradient.
                float wv = weight * v;
                float vg[DIM];
                int moment = MOMENT;
                #pragma unroll
                for (int d = 0; d < DIM; ++d) {
                    vg[d] = v * grad_weight[d];
                    if (save_backward)
                        partial[WEIGHTED_SUM + d] += vg[d];
                    #pragma unroll
                    for (int e = d; e < DIM; ++e)
                        partial[moment++] += vg[d] * r[e];
                }
                const float4 *remote_state = reinterpret_cast<const float4 *>(tile + j * CHANNELS);
                #pragma unroll
                for (int q = 0; q < CHANNELS / 4; ++q) {
                    float4 packed = remote_state[q];
                    float values[4] = {packed.x, packed.y, packed.z, packed.w};
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        int c = 4 * q + k;
                        partial[BLUR + c] += values[k] * wv;
                        float difference = values[k] - local_state[c];
                        #pragma unroll
                        for (int d = 0; d < DIM; ++d)
                            partial[GRADIENT + c * DIM + d] += difference * vg[d];
                    }
                }
            });
        reduce_lanes<PARTIALS>(partial, share);
        reduce_teams<PARTIALS, Layout>(partial, shared, team, group, lane);
        if constexpr (NEIGHBOR_WARPS == 1)
            __syncthreads();  // reduce_teams has no barrier; staging aliases every warp's tile.
        // Team 0's first lane of each particle owns the outputs; the other
        // lanes and teams join the barriers and copies.
        bool owner = team == 0 && valid && sub == 0;
        unsigned int rows = min(32u >> shift, info.particle_count - offset);
        unsigned int first_row = start + offset;
        offset += 32u >> shift;

        if (owner) {
            float matrix[DIM * DIM];
            int moment = MOMENT;
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
                #pragma unroll
                for (int e = d; e < DIM; ++e)
                    matrix[d * DIM + e] = matrix[e * DIM + d] = partial[moment++];
            safe_inverse_symmetric<DIM>(matrix, MOMENT_INVERSE_TOLERANCE);
            // Moment correction changes the forward value only: gradient_hybrid
            // uses raw + (raw @ inverse - raw).detach(). Its adjoint is in backward.
            // The corrected gradient replaces the raw one in the partials.
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                float corrected[DIM] = {0.0f};
                #pragma unroll
                for (int d = 0; d < DIM; ++d) {
                    #pragma unroll
                    for (int e = 0; e < DIM; ++e)
                        corrected[d] += partial[GRADIENT + c * DIM + e] * matrix[e * DIM + d];
                    corrected[d] *= scale;
                }
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                    partial[GRADIENT + c * DIM + d] = corrected[d];
            }
            if (group == 0 && save_backward) {
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                    gradient_sum[particle * DIM + d] = partial[WEIGHTED_SUM + d];
            }
        }

        // Stage both output rows at once: perception rows first, then (when
        // saved for backward) the corrected gradients before normalization,
        // rows [FEATURES][DIM], behind the widest perception rows.
        constexpr unsigned int WIDTH = FEATURES * DIM;
        constexpr unsigned int GRADIENT_STAGE = 32 * ((FEATURES * (2 + DIM) + DIM) | 1);
        bool stage_gradients = save_backward && normalize_state_grad;
        if (owner) {
            float *gradient_row = shared + GRADIENT_STAGE + slot * staging_stride(WIDTH);
            // Perception rows [state(F), blur(F), gradient(F*DIM), optional density gradient(DIM)].
            float *row = shared + slot * staging_stride(output_dim);
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                unsigned int channel = first_channel + c;
                float *corrected = &partial[GRADIENT + c * DIM];
                if (stage_gradients) {
                    #pragma unroll
                    for (int d = 0; d < DIM; ++d)
                        gradient_row[channel * DIM + d] = corrected[d];
                }
                if (normalize_state_grad)
                    log_normalize<DIM>(corrected);
                row[channel] = local_state[c];
                row[FEATURES + channel] = partial[BLUR + c];
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                    row[2 * FEATURES + channel * DIM + d] = corrected[d];
            }
            if (group == 0 && perceive_density_grad) {
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
                    row[FEATURES * (2 + DIM) + d] = grad_rho[d];
            }
        }
        __syncthreads();
        if (stage_gradients)
            store_rows<Layout::THREADS>(gradient_values + first_row * WIDTH, shared + GRADIENT_STAGE, rows, WIDTH);
        store_rows<Layout::THREADS>(perception + first_row * output_dim, shared, rows, output_dim);
        __syncthreads();  // The next particle chunk or grid block refills the tiles.
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
                                  THREAD_CHANNELS<FEATURES>, BACKWARD_RESIDENT_WARPS, 32 * (FEATURES | 1),
                                  BACKWARD_TILE_BUFFERS>;

template <int DIM, int FEATURES, BoundaryCondition BC>
__global__ void __launch_bounds__(BackwardLayout<DIM, FEATURES>::THREADS,
                                  BackwardLayout<DIM, FEATURES>::MIN_BLOCKS) backward(
    const ParticlePosition<DIM> *__restrict__ positions,
    const float *__restrict__ volume, const float *__restrict__ incoming,
    float *__restrict__ grad_state, const uint32_t *__restrict__ bin_offsets,
    const BlockInfo *__restrict__ block_info,
    GridConfig<DIM, BC> config)
{
    static_assert(FEATURES == 4 || FEATURES == 8 || FEATURES == 16 || FEATURES == 32,
                  "Native feature counts are vectorizable powers of two");
    using Layout = BackwardLayout<DIM, FEATURES>;
    constexpr int CHANNELS = Layout::CHANNELS;
    constexpr int TILE = Layout::TILE;
    constexpr int STRIDE = (DIM + 1) * FEATURES;  // Global incoming row per particle.
    constexpr int PAYLOAD = (DIM + 1) * CHANNELS;  // This channel group's share.
    // Persistent loop over grid blocks: the active records are a contiguous
    // prefix of block_info, so the first zero-count sentinel ends the work of
    // this CUDA block without scheduling the empty tail (large in sparse 3D).
    unsigned int max_blocks = config.max_blocks_per_batch * config.batch_size;
    for (unsigned int block = blockIdx.x; block < max_blocks; block += gridDim.x) {
    BlockInfo info = block_info[block];
    if (info.particle_count == 0)
        break;
    __shared__ __align__(16) float shared[Layout::SHARED_FLOATS];

    int cell[DIM];
    config.hash2cell(info.cell_idx - info.batch_idx * config.cell_count, cell);
    unsigned int share_cap = lane_share_cap(info.candidates_per_row);
    unsigned int start = bin_offsets[info.cell_idx] + info.offset_in_cell;
    unsigned int lane = threadIdx.x & 31;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int group = warp % Layout::SPLIT;
    unsigned int team = warp / Layout::SPLIT;
    unsigned int first_channel = group * CHANNELS;
    // Per buffer: [cotangents][positions].
    float *tiles = shared + warp * Layout::WARP_FLOATS;
    constexpr int POSITION_OFFSET = TILE * PAYLOAD;
    float h2 = config.eps * config.eps;

    for (unsigned int offset = 0; offset < info.particle_count;) {
        unsigned int shift = min(lane_share_shift(info.particle_count - offset), share_cap);
        unsigned int share = 1u << shift, slot = lane >> shift, sub = lane & (share - 1);
        bool valid = offset + slot < info.particle_count;
        unsigned int particle = start + offset + slot;
        ParticlePosition<DIM> position{};
        if (valid)
            position = positions[particle];
        float result[CHANNELS] = {0.0f};
        for_each_neighbor<TILE, Layout::TILE_FLOATS, Layout::BUFFERS, POSITION_OFFSET>(
            config, cell, info.batch_idx, bin_offsets, team, valid, sub, shift, position, tiles, h2,
            [&](unsigned int remote, unsigned int count, float *tile) {
                if (lane < count)
                    copy_async_position<DIM>(reinterpret_cast<ParticlePosition<DIM> *>(tile + POSITION_OFFSET) + lane,
                                             positions + remote + lane);
                // Tile layout [neighbor][component][channel of this group], copied
                // as float4 chunks: chunk k of a row is component k / (CHANNELS / 4).
                constexpr unsigned int QUADS = PAYLOAD / 4;
                constexpr unsigned int GROUP_QUADS = CHANNELS / 4;
                #pragma unroll 4
                for (unsigned int k = lane; k < count * QUADS; k += 32) {
                    unsigned int j = k / QUADS, q = k % QUADS;
                    copy_async16(tile + j * PAYLOAD + q * 4,
                                 incoming + (remote + j) * STRIDE + (q / GROUP_QUADS) * FEATURES
                                          + first_channel + (q % GROUP_QUADS) * 4);
                }
            },
            [&](unsigned int j, const ParticlePosition<DIM> &r, float d2, const float *tile) {
                float weight, grad_weight[DIM];
                support_weights(config, r, d2, h2, weight, grad_weight);
                const float *cotangent = tile + j * PAYLOAD;
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
            });
        reduce_lanes<CHANNELS>(result, share);
        reduce_teams<CHANNELS, Layout>(result, shared, team, group, lane);
        if constexpr (NEIGHBOR_WARPS == 1)
            __syncthreads();  // reduce_teams has no barrier; staging aliases every warp's tile.
        // Team 0's first lane of each particle stages its sums; all threads
        // then add volume_i * sum to the rows backward_prepare wrote, coalesced,
        // as one fused multiply-add (the rounding of the former per-lane
        // `+= volume * sum`).
        if (team == 0 && valid && sub == 0) {
            float *row = shared + slot * staging_stride(FEATURES);
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c)
                row[first_channel + c] = result[c];
        }
        __syncthreads();
        unsigned int rows = min(32u >> shift, info.particle_count - offset);
        float *destination = grad_state + (start + offset) * FEATURES;
        for (unsigned int k = threadIdx.x; k < rows * FEATURES; k += Layout::THREADS) {
            unsigned int row = k / FEATURES;
            destination[k] = fmaf(volume[start + offset + row],
                                  shared[row * staging_stride(FEATURES) + k % FEATURES], destination[k]);
        }
        offset += 32u >> shift;
        __syncthreads();  // The next particle chunk or grid block refills the tiles.
    }
    }
}

} // namespace fused_perception
