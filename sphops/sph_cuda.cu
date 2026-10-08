#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <stdexcept>
#include <map>
#include "utils.cuh"
#include "fused_perception.cuh"

namespace nb = nanobind;
using namespace nb::literals;

// Type aliases for nd-arrays
using FloatArray1D = nb::ndarray<float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using FloatArray2D = nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig, nb::device::cuda>;
using FloatArray3D = nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig, nb::device::cuda>;
using FloatArray4D = nb::ndarray<float, nb::shape<-1, -1, -1, -1>, nb::c_contig, nb::device::cuda>;
using IntArray1D = nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using IntArray2D = nb::ndarray<int32_t, nb::shape<-1, -1>, nb::c_contig, nb::device::cuda>;
using BlockInfoArray = nb::ndarray<int32_t, nb::shape<-1, 4>, nb::c_contig, nb::device::cuda>;

#define cuda_check(call)                                                              \
    do                                                                                \
    {                                                                                 \
        cudaError_t error = call;                                                     \
        if (error != cudaSuccess)                                                     \
        {                                                                             \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - " +               \
                                     cudaGetErrorString(error));                      \
        }                                                                             \
    } while (0)

// The two counting-sort prefix sums (cell counts and blocks per cell) run as
// one scan over zipped pairs.
using CounterPair = thrust::tuple<uint32_t, uint32_t>;
struct CounterPairSum
{
    __host__ __device__ CounterPair operator()(const CounterPair &a, const CounterPair &b) const
    {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b), thrust::get<1>(a) + thrust::get<1>(b));
    }
};

// Counting pass of the counting sort. Each particle's arrival rank within its
// cell is kept in the permutation buffer, so the scatter pass needs no second
// set of atomics (the order within a cell is arbitrary in either case).
template <int DIM, BoundaryCondition BC>
__global__ void grid_count_particles(
    const ParticlePosition<DIM> * __restrict__ positions,
    uint32_t * __restrict__ cell_counts,
    uint32_t * __restrict__ blocks_per_cell,
    uint32_t * __restrict__ rank_in_cell,
    GridConfig<DIM, BC> config)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    uint32_t idx = batch_idx * config.particle_count + tid;
    if (tid >= config.particle_count)
        return;
    if (batch_idx >= config.batch_size)
        return;

    uint32_t cell_idx = batch_idx * config.cell_count + config.cell_hash(positions[idx]);

    uint32_t prev_count = atomicAdd(&cell_counts[cell_idx], 1U);
    rank_in_cell[idx] = prev_count;
    if (prev_count % config.MAX_PARTICLES_PER_BLOCK == 0)
        atomicAdd(&blocks_per_cell[cell_idx], 1U);
}

// Scatter pass: permutation[idx] holds the rank from grid_count_particles on
// entry and the particle's binned index within its batch on exit.
template <int DIM, BoundaryCondition BC>
__global__ void grid_compute_permutation(
    const ParticlePosition<DIM> * __restrict__ positions,
    const uint32_t * __restrict__ bin_offsets,
    uint32_t * __restrict__ permutation,
    GridConfig<DIM, BC> config)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    uint32_t idx = batch_idx * config.particle_count + tid;
    if (tid >= config.particle_count)
        return;
    if (batch_idx >= config.batch_size)
        return;

    uint32_t hash = config.cell_hash(positions[idx]);
    permutation[idx] += bin_offsets[batch_idx * config.cell_count + hash] - batch_idx * config.particle_count;
}

// Copies one row of `width` floats, as float4 when width is a multiple of 4.
__device__ __forceinline__ void copy_row(float *__restrict__ dst, const float *__restrict__ src, uint32_t width)
{
    if ((width & 3) == 0)
    {
        for (uint32_t f = 0; f < width; f += 4)
            reinterpret_cast<float4 *>(dst)[f >> 2] = reinterpret_cast<const float4 *>(src)[f >> 2];
    }
    else
    {
        for (uint32_t f = 0; f < width; ++f)
            dst[f] = src[f];
    }
}

// Scatter pass of the fused path: the binned index of grid_compute_permutation,
// and the particle copied to it as apply_permutation does, in one kernel.
template <int DIM, BoundaryCondition BC>
__global__ void grid_scatter_and_bin(
    const ParticlePosition<DIM> * __restrict__ positions,
    const float * __restrict__ input_state,
    const float * __restrict__ input_mass,
    const uint32_t * __restrict__ bin_offsets,
    uint32_t * __restrict__ permutation,
    ParticlePosition<DIM> * __restrict__ output_positions,
    float * __restrict__ output_state,
    float * __restrict__ output_mass,
    uint32_t feature_dim,
    GridConfig<DIM, BC> config)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    uint32_t idx = batch_idx * config.particle_count + tid;
    if (tid >= config.particle_count || batch_idx >= config.batch_size)
        return;

    ParticlePosition<DIM> position = positions[idx];
    uint32_t hash = config.cell_hash(position);
    uint32_t binned = permutation[idx] + bin_offsets[batch_idx * config.cell_count + hash] - batch_idx * config.particle_count;
    permutation[idx] = binned;
    uint32_t permuted_idx = binned + batch_idx * config.particle_count;
    output_positions[permuted_idx] = position;
    if (input_mass != nullptr)  // Optional: unit masses need no binning.
        output_mass[permuted_idx] = input_mass[idx];
    copy_row(&output_state[permuted_idx * feature_dim], &input_state[idx * feature_dim], feature_dim);
}

template <int DIM, BoundaryCondition BC>
__global__ void apply_permutation(
    const ParticlePosition<DIM> * __restrict__ input_positions,
    const float * __restrict__ input_state,
    const float * __restrict__ input_mass,
    const uint32_t * __restrict__ permutation,
    ParticlePosition<DIM> * __restrict__ output_positions,
    float * __restrict__ output_state,
    float * __restrict__ output_mass,
    uint32_t feature_dim,
    GridConfig<DIM, BC> config)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    if (tid >= config.particle_count)
        return;
    if (batch_idx >= config.batch_size)
        return;

    uint32_t idx = batch_idx * config.particle_count + tid;
    uint32_t permuted_idx = permutation[idx] + batch_idx * config.particle_count;
    output_positions[permuted_idx] = input_positions[idx];
    if (input_mass != nullptr)  // Optional: unit masses need no binning.
        output_mass[permuted_idx] = input_mass[idx];
    copy_row(&output_state[permuted_idx * feature_dim], &input_state[idx * feature_dim], feature_dim);
}

// Gather cotangents back to the original particle order. Position and mass
// cotangents are optional (nullptr). An optional second binned state cotangent
// is summed in the same pass, as autograd would accumulate the two.
template <int DIM, BoundaryCondition BC>
__global__ void backward_apply_permutation(
    const ParticlePosition<DIM> * __restrict__ grad_output_positions,
    const float * __restrict__ grad_output_state,
    const float * __restrict__ grad_output_mass,
    const uint32_t * __restrict__ permutation,
    ParticlePosition<DIM> * __restrict__ grad_input_positions,
    float * __restrict__ grad_input_state,
    float * __restrict__ grad_input_mass,
    uint32_t feature_dim,
    GridConfig<DIM, BC> config,
    const float * __restrict__ grad_output_state_addend = nullptr)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    if (tid >= config.particle_count)
        return;
    if (batch_idx >= config.batch_size)
        return;

    uint32_t idx = batch_idx * config.particle_count + tid;

    uint32_t permuted_idx = permutation[idx] + batch_idx * config.particle_count;

    if (grad_output_positions != nullptr)
        grad_input_positions[idx] = grad_output_positions[permuted_idx];
    if (grad_output_mass != nullptr)
        grad_input_mass[idx] = grad_output_mass[permuted_idx];

    if (grad_output_state == nullptr)
        return;
    const float *src = &grad_output_state[permuted_idx * feature_dim];
    float *dst = &grad_input_state[idx * feature_dim];
    if (grad_output_state_addend != nullptr)
    {
        const float *addend = &grad_output_state_addend[permuted_idx * feature_dim];
        if ((feature_dim & 3) == 0)
        {
            for (uint32_t f = 0; f < feature_dim; f += 4)
            {
                float4 a = reinterpret_cast<const float4 *>(src)[f >> 2];
                float4 b = reinterpret_cast<const float4 *>(addend)[f >> 2];
                reinterpret_cast<float4 *>(dst)[f >> 2] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
        }
        else
            for (uint32_t f = 0; f < feature_dim; ++f)
                dst[f] = src[f] + addend[f];
    }
    else
        copy_row(dst, src, feature_dim);
}

// One entry per grid block: its cell, offset within the cell, particle count and batch.
template <int DIM, BoundaryCondition BC>
__global__ void assign_block_info(
    const uint32_t * __restrict__ cell_counts,   // Shape [batch_size * cell_count]
    const uint32_t * __restrict__ block_offsets, // Cumulative sum of blocks per cell (batch_size * cell_count + 1)
    BlockInfo * __restrict__ block_info,         // Output: block information # Shape [batch_size * max_blocks_per_batch]
    GridConfig<DIM, BC> config)
{
    uint32_t cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;

    if (batch_idx >= config.batch_size)
        return;

    // The launch covers max_blocks records; the unused tail after the last
    // grid block becomes zero-count sentinels here instead of by a memset.
    uint32_t total_blocks = block_offsets[config.batch_size * config.cell_count];
    uint32_t max_blocks = config.max_blocks_per_batch * config.batch_size;
    uint32_t threads = gridDim.x * blockDim.x * config.batch_size;
    for (uint32_t i = total_blocks + (batch_idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
         i < max_blocks; i += threads)
        block_info[i] = BlockInfo{0u, 0u, 0u, 0u};

    if (cell_idx >= config.cell_count)
        return;

    uint32_t global_cell_idx = batch_idx * config.cell_count + cell_idx;
    uint32_t particles_in_cell = cell_counts[global_cell_idx];

    if (particles_in_cell == 0)
        return;

    // Get the starting block index for this cell within the batch
    uint32_t batch_offset_base = batch_idx * config.cell_count;
    uint32_t block_base_idx = block_offsets[batch_offset_base + cell_idx];
    uint32_t blocks_needed = (particles_in_cell + config.MAX_PARTICLES_PER_BLOCK - 1) / config.MAX_PARTICLES_PER_BLOCK;

    for (uint32_t block_num = 0; block_num < blocks_needed; ++block_num)
    {
        // Fill in the BlockInfo structure
        BlockInfo info;
        info.cell_idx = global_cell_idx;                                  // Use global cell index
        info.offset_in_cell = block_num * config.MAX_PARTICLES_PER_BLOCK; // Fix: this should be byte offset, not block number
        info.particle_count = min(config.MAX_PARTICLES_PER_BLOCK,
                                  particles_in_cell - block_num * config.MAX_PARTICLES_PER_BLOCK);
        info.batch_idx = batch_idx; // Store the batch index

        // Store the BlockInfo in the output array
        block_info[block_base_idx + block_num] = info;
    }
}

// Splits a flat index over rows of `width` values into (row, column), with a
// shift/mask path for power-of-two widths.
struct RowSplit
{
    uint32_t width;
    uint32_t shift = 0;
    bool pow2;

    __device__ explicit RowSplit(uint32_t width) : width(width), pow2((width & (width - 1)) == 0)
    {
        while ((1U << shift) < width)
            shift++;
    }
    __device__ uint32_t row(uint32_t k) const { return pow2 ? k >> shift : k / width; }
    __device__ uint32_t col(uint32_t k) const { return pow2 ? k & (width - 1) : k % width; }
};

enum class ComputationType
{
    Count,           // 0
    Density,         // 1
    DensityGradient, // 2
    MomentMatrix,    // 3
    Blur,            // 4
    Gradient         // 5
};

template <int DIM, ComputationType COMP_TYPE, int STRIDE = 128,
          BoundaryCondition BC = BoundaryCondition::PERIODIC,
          bool COMPUTE_DS_ONLY = false>
__global__ void backward_kernel(
    const ParticlePosition<DIM> * __restrict__ position_in, // Shape (batch_size, N, DIM)
    const float * __restrict__ rho_in,                      // Input density tensor, Shape (batch_size, N)
    const float * __restrict__ mass_in,                     // Input mass tensor, Shape (batch_size, N)
    const float * __restrict__ state_in,                    // Input state tensor, Shape (batch_size, N, feature_dim)
    const float * __restrict__ dL_rho_out,                  // Gradient of loss w.r.t. output density, Shape (batch_size, N)
    const float * __restrict__ dL_count_out,                // Gradient of loss w.r.t. output count, Shape (batch_size, N)
    const float * __restrict__ dL_blur_state_out,           // Gradient of loss w.r.t. output blurred state, Shape (batch_size, N, feature_dim)
    const float * __restrict__ dL_grad_state_out,           // Gradient of loss w.r.t. output state gradient, Shape (batch_size, N, feature_dim, DIM)
    const float * __restrict__ dL_grad_density_out,         // Gradient of loss w.r.t. output density gradient, Shape (batch_size, N, DIM)
    const float * __restrict__ dL_moment_matrix_out,        // Gradient of loss w.r.t. output moment matrix, Shape (batch_size, N, DIM, DIM)
    ParticlePosition<DIM> * __restrict__ dL_position_in,    // Gradient of loss w.r.t. input position, Shape (batch_size, N, DIM)
    float * __restrict__ dL_rho_in,                         // Gradient of loss w.r.t. input density, Shape (batch_size, N)
    float * __restrict__ dL_state_in,                       // Gradient of loss w.r.t. input state, Shape (batch_size, N, feature_dim)
    uint32_t feature_dim,                     // Number of features in state
    const uint32_t * __restrict__ bin_offsets, // Shape (batch_size * cell_count + 1)
    const BlockInfo * __restrict__ block_info, // Block information array
    const GridConfig<DIM, BC> config)
{
    // BlockInfo belongs to the saved snapshot. The grid's block offsets are
    // scratch storage and may already describe a later rollout step.
    BlockInfo b_info = block_info[blockIdx.x];
    if (b_info.particle_count == 0)
        return;

    __shared__ ParticlePosition<DIM> local_pos[STRIDE];
    __shared__ ParticlePosition<DIM> neighbor_pos[STRIDE];
    extern __shared__ float smem[];

    float *dL_dx_buffer, *dL_drho_buffer, *dL_ds_buffer;
    float *dL_local, *dL_neighbor;
    float *local_state, *neighbor_state;
    float *local_density_rcp, *neighbor_density_rcp;
    float *local_mass, *neighbor_mass;
    const float *incoming_grad_buffer;
    uint32_t incoming_grad_buffer_dim;
    float default_mass = config.default_mass;

    if constexpr (COMP_TYPE == ComputationType::Count)
        return; // No gradients to compute for count
    else if constexpr (COMP_TYPE == ComputationType::Density)
    {
        if constexpr (COMPUTE_DS_ONLY)
            return;                          // No gradients to compute if not required
        dL_local = &smem[0];                 // STRIDE
        dL_neighbor = &dL_local[STRIDE];     // STRIDE
        dL_dx_buffer = &dL_neighbor[STRIDE]; // STRIDE * DIM
        local_state = neighbor_state = nullptr;
        local_density_rcp = neighbor_density_rcp = nullptr;
        dL_drho_buffer = dL_ds_buffer = nullptr;
        local_mass = &dL_dx_buffer[STRIDE * DIM]; // STRIDE
        neighbor_mass = &local_mass[STRIDE];      // STRIDE
        incoming_grad_buffer = dL_rho_out;
        incoming_grad_buffer_dim = 1;
    }
    else if constexpr (COMP_TYPE == ComputationType::Blur)
    {
        if constexpr (COMPUTE_DS_ONLY)
        {
            dL_neighbor = &smem[0];                                 // feature_dim * STRIDE
            local_density_rcp = &dL_neighbor[feature_dim * STRIDE]; // STRIDE
            neighbor_density_rcp = nullptr;
            local_mass = &local_density_rcp[STRIDE]; // STRIDE
            neighbor_mass = nullptr;
            dL_ds_buffer = &local_mass[STRIDE]; // feature_dim * STRIDE
            dL_local = dL_dx_buffer = dL_drho_buffer = nullptr;
            local_state = neighbor_density_rcp = nullptr;
            neighbor_state = nullptr;
        }
        else
        {
            dL_local = &smem[0];                                       // feature_dim * STRIDE
            dL_neighbor = &dL_local[feature_dim * STRIDE];             // feature_dim * STRIDE
            local_state = &dL_neighbor[feature_dim * STRIDE];          // feature_dim * STRIDE
            neighbor_state = &local_state[feature_dim * STRIDE];       // feature_dim * STRIDE
            local_density_rcp = &neighbor_state[feature_dim * STRIDE]; // STRIDE
            neighbor_density_rcp = &local_density_rcp[STRIDE];         // STRIDE
            local_mass = &neighbor_density_rcp[STRIDE];                // STRIDE
            neighbor_mass = &local_mass[STRIDE];                       // STRIDE
            dL_ds_buffer = &neighbor_mass[STRIDE];                     // feature_dim * STRIDE
            dL_dx_buffer = &dL_ds_buffer[feature_dim * STRIDE];        // DIM * STRIDE
            dL_drho_buffer = &dL_dx_buffer[DIM * STRIDE];              // STRIDE
        }
        incoming_grad_buffer = dL_blur_state_out;
        incoming_grad_buffer_dim = feature_dim;
    }
    else if constexpr (COMP_TYPE == ComputationType::Gradient)
    {
        if constexpr (COMPUTE_DS_ONLY)
        {
            dL_local = &smem[0];                                          // feature_dim * DIM * STRIDE
            dL_neighbor = &dL_local[feature_dim * DIM * STRIDE];          // feature_dim * DIM * STRIDE
            local_density_rcp = &dL_neighbor[feature_dim * DIM * STRIDE]; // STRIDE
            neighbor_density_rcp = &local_density_rcp[STRIDE];            // STRIDE
            local_mass = &neighbor_density_rcp[STRIDE];                   // STRIDE
            neighbor_mass = &local_mass[STRIDE];                          // STRIDE
            dL_ds_buffer = &neighbor_mass[STRIDE];                        // feature_dim * STRIDE
            dL_dx_buffer = nullptr;
            dL_drho_buffer = nullptr;
            local_state = neighbor_state = nullptr;
        }
        else
        {
            dL_local = &smem[0];                                       // feature_dim * DIM * STRIDE
            dL_neighbor = &dL_local[feature_dim * DIM * STRIDE];       // feature_dim * DIM * STRIDE
            local_state = &dL_neighbor[feature_dim * DIM * STRIDE];    // feature_dim * STRIDE
            neighbor_state = &local_state[feature_dim * STRIDE];       // feature_dim * STRIDE
            local_density_rcp = &neighbor_state[feature_dim * STRIDE]; // STRIDE
            neighbor_density_rcp = &local_density_rcp[STRIDE];         // STRIDE
            local_mass = &neighbor_density_rcp[STRIDE];                // STRIDE
            neighbor_mass = &local_mass[STRIDE];                       // STRIDE
            dL_ds_buffer = &neighbor_mass[STRIDE];                     // feature_dim * STRIDE
            dL_dx_buffer = &dL_ds_buffer[feature_dim * STRIDE];        // DIM * STRIDE
            dL_drho_buffer = &dL_dx_buffer[DIM * STRIDE];              // STRIDE
        }
        incoming_grad_buffer = dL_grad_state_out;
        incoming_grad_buffer_dim = feature_dim * DIM;
    }
    else if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
    {
        if constexpr (COMPUTE_DS_ONLY)
            return;                                           // No gradients to compute if not required
        dL_local = &smem[0];                                  // DIM * DIM * STRIDE
        dL_neighbor = &dL_local[DIM * DIM * STRIDE];          // DIM * DIM * STRIDE
        local_density_rcp = &dL_neighbor[DIM * DIM * STRIDE]; // STRIDE
        neighbor_density_rcp = &local_density_rcp[STRIDE];    // STRIDE
        local_mass = &neighbor_density_rcp[STRIDE];           // STRIDE
        neighbor_mass = &local_mass[STRIDE];                  // STRIDE
        dL_dx_buffer = &neighbor_mass[STRIDE];                // DIM * STRIDE
        dL_drho_buffer = &dL_dx_buffer[DIM * STRIDE];         // STRIDE
        dL_ds_buffer = nullptr;
        local_state = neighbor_state = nullptr;
        incoming_grad_buffer = dL_moment_matrix_out;
        incoming_grad_buffer_dim = DIM * DIM;
    }
    else if constexpr (COMP_TYPE == ComputationType::DensityGradient)
    {
        if constexpr (COMPUTE_DS_ONLY)
            return;                                // No gradients to compute if not required
        dL_local = &smem[0];                       // DIM * STRIDE
        dL_neighbor = &dL_local[DIM * STRIDE];     // DIM * STRIDE
        dL_dx_buffer = &dL_neighbor[DIM * STRIDE]; // DIM * STRIDE
        local_mass = &dL_dx_buffer[DIM * STRIDE];  // STRIDE
        neighbor_mass = &local_mass[STRIDE];       // STRIDE
        local_state = neighbor_state = nullptr;
        local_density_rcp = neighbor_density_rcp = nullptr;
        dL_drho_buffer = dL_ds_buffer = nullptr;
        incoming_grad_buffer = dL_grad_density_out;
        incoming_grad_buffer_dim = DIM;
    }
    else
    {
        return; // Unsupported computation type
    }

    const RowSplit features(feature_dim);
    const RowSplit grads(incoming_grad_buffer_dim);

    uint32_t batch_idx = b_info.batch_idx;
    int cell[DIM];
    config.hash2cell(b_info.cell_idx % config.cell_count, cell);

    // Batch-local indices of this grid block's particles.
    uint32_t local_start = bin_offsets[b_info.cell_idx] - batch_idx * config.particle_count + b_info.offset_in_cell;
    uint32_t local_end = bin_offsets[b_info.cell_idx + 1] - batch_idx * config.particle_count;
    local_end = min(local_end, local_start + b_info.particle_count);
    uint32_t local_count = local_end - local_start;
    float eps2 = config.eps * config.eps;

    ParticlePosition<DIM> *dL_dx_global;
    float *dL_drho_global, *dL_ds_global;

    // Local particles in chunks of STRIDE; each chunk visits all neighbor tiles.
    for (uint32_t local_offset = 0; local_offset < local_count; local_offset += STRIDE)
    {
        uint32_t local_chunk = min(STRIDE, local_count - local_offset);

        // Local particle positions
        for (uint32_t k = threadIdx.x; k < local_chunk; k += blockDim.x)
            local_pos[k] = read_position<DIM>(position_in, config.particle_count, batch_idx,
                                              local_start + local_offset + k);

        // Load Backprop gradients for local particles
        if constexpr (!COMPUTE_DS_ONLY || COMP_TYPE != ComputationType::Blur)
        {
            for (uint32_t k = threadIdx.x; k < local_chunk * incoming_grad_buffer_dim; k += blockDim.x)
            {
                dL_local[k] = read_state(incoming_grad_buffer, config.particle_count, incoming_grad_buffer_dim,
                                         batch_idx, local_start + local_offset + grads.row(k), grads.col(k));
            }
        }
        // Each thread loads local state if needed
        if constexpr (!COMPUTE_DS_ONLY)
        {
            if constexpr (COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::Blur)
            {
                for (uint32_t k = threadIdx.x; k < local_chunk * feature_dim; k += blockDim.x)
                {
                    local_state[k] = read_state(state_in, config.particle_count, feature_dim,
                                                batch_idx, local_start + local_offset + features.row(k), features.col(k));
                }
            }
        }
        // Each thread loads local mass if needed
        if constexpr (COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::MomentMatrix ||
                      COMP_TYPE == ComputationType::DensityGradient || COMP_TYPE == ComputationType::Density)
        {
            for (uint32_t k = threadIdx.x; k < local_chunk; k += blockDim.x)
            {
                uint32_t idx = local_start + local_offset + k;
                local_mass[k] = read_mass(mass_in, config.particle_count, batch_idx, idx, default_mass);
            }
        }
        // Each thread loads local density if needed
        if constexpr (COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::MomentMatrix)
        {
            for (uint32_t k = threadIdx.x; k < local_chunk; k += blockDim.x)
            {
                uint32_t idx = local_start + local_offset + k;
                float rho_i = read_state(rho_in, config.particle_count, 1, batch_idx, idx, 0);
                local_density_rcp[k] = local_mass[k] / rho_i; // v_i = m_i / rho_i => 1/v_i = rho_i / m_i
                
                // For blur/gradient/moment matrix backward paths, cache (v_i^2 / m_i)
                // once per local particle and reuse it in the inner interaction loops.
                local_mass[k] = local_mass[k] / (rho_i * rho_i); // m_i / (rho_i^2) for later use in gradients
            }
        }

        // Global output pointers for gradients w.r.t. input to be filled
        if constexpr (!COMPUTE_DS_ONLY)
        {
            dL_dx_global = &dL_position_in[batch_idx * config.particle_count + local_start + local_offset];
            if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::MomentMatrix)
            {
                dL_drho_global = &dL_rho_in[batch_idx * config.particle_count + local_start + local_offset];
            }
        }
        if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient)
        {
            dL_ds_global = &dL_state_in[(batch_idx * config.particle_count + local_start + local_offset) * feature_dim];
        }

        // Zero out local gradient accumulators
        if constexpr (!COMPUTE_DS_ONLY)
        {
            for (uint32_t k = threadIdx.x; k < local_chunk * DIM; k += blockDim.x)
                dL_dx_buffer[k] = 0.0f;
        }

        if constexpr (!COMPUTE_DS_ONLY)
        {
            if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::MomentMatrix)
            {
                for (uint32_t k = threadIdx.x; k < local_chunk; k += blockDim.x)
                    dL_drho_buffer[k] = 0.0f;
            }
        }

        if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient)
        {
            for (uint32_t k = threadIdx.x; k < local_chunk * feature_dim; k += blockDim.x)
                dL_ds_buffer[k] = 0.0f;
        }

        // Same spans as GridConfig::neighbor_span, computed inline: in these
        // block-wide kernels the inline form compiles to faster code.
        for (int row = 0; row < GridConfig<DIM, BC>::NEIGHBOR_ROWS; ++row)
        {
            int neighbor[DIM];
            if (!config.neighbor_row(cell, row, neighbor))
                continue;
            for (int wrap = 0; wrap < GridConfig<DIM, BC>::WRAP_CHUNKS; ++wrap)
            {
                int first_x, last_x;
                if (!config.row_x_range(cell, wrap, first_x, last_x))
                    continue;
                neighbor[0] = first_x;
                uint32_t first = config.cell2hash(neighbor);
                neighbor[0] = last_x;
                uint32_t last = config.cell2hash(neighbor);
                // Batch-local indices, as read_position and read_state expect.
                uint32_t remote_start = bin_offsets[batch_idx * config.cell_count + first] - batch_idx * config.particle_count;
                uint32_t remote_end = bin_offsets[batch_idx * config.cell_count + last + 1] - batch_idx * config.particle_count;
                uint32_t remote_count = remote_end - remote_start;
                if (remote_count == 0)
                    continue;

                for (uint32_t remote_offset = 0; remote_offset < remote_count; remote_offset += STRIDE)
                {
                    uint32_t remote_chunk = min(STRIDE, remote_count - remote_offset);

                    // Each thread loads a few neighbor particles into shared memory
                    for (uint32_t k = threadIdx.x; k < remote_chunk; k += blockDim.x)
                        neighbor_pos[k] = read_position<DIM>(position_in, config.particle_count, batch_idx,
                                                             remote_start + remote_offset + k);

                    // Each thread loads neighbor state if needed
                    if constexpr (!COMPUTE_DS_ONLY)
                    {
                        if constexpr (COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::Blur)
                        {
                            for (uint32_t k = threadIdx.x; k < remote_chunk * feature_dim; k += blockDim.x)
                            {
                                neighbor_state[k] = read_state(state_in, config.particle_count, feature_dim, batch_idx,
                                                               remote_start + remote_offset + features.row(k), features.col(k));
                            }
                        }
                    }

                    // Each thread loads neighbor mass/density if needed
                    if constexpr (COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::MomentMatrix ||
                                  COMP_TYPE == ComputationType::DensityGradient || COMP_TYPE == ComputationType::Density)
                    {
                        if constexpr (!COMPUTE_DS_ONLY || COMP_TYPE != ComputationType::Blur)
                        {
                            for (uint32_t k = threadIdx.x; k < remote_chunk; k += blockDim.x)
                            {
                                uint32_t idx = remote_start + remote_offset + k;
                                neighbor_mass[k] = read_mass(mass_in, config.particle_count, batch_idx, idx, default_mass);
                                if constexpr (COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::MomentMatrix) 
                                {
                                    float rho_j = read_state(rho_in, config.particle_count, 1, batch_idx, idx, 0);
                                    neighbor_density_rcp[k] = neighbor_mass[k] / rho_j;
                                }
                            }
                        }
                    }

                    // Each thread loads the backprop gradients for neighbor particles
                    for (uint32_t k = threadIdx.x; k < remote_chunk * incoming_grad_buffer_dim; k += blockDim.x)
                    {
                        dL_neighbor[k] = read_state(incoming_grad_buffer, config.particle_count, incoming_grad_buffer_dim,
                                                    batch_idx, remote_start + remote_offset + grads.row(k), grads.col(k));
                    }

                    __syncthreads();

                    if constexpr (COMP_TYPE == ComputationType::Density)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk; i += blockDim.x)
                        {
                            ParticlePosition<DIM> p_i = local_pos[i];
                            float grad_i = dL_local[i];
                            float m_i = local_mass[i];

                            float output[DIM] = {0.0f};
                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                float grad_j = dL_neighbor[j];
                                ParticlePosition<DIM> p_j = neighbor_pos[j];
                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    float m_j = neighbor_mass[j];
                                    float grad_coef = grad_i * m_j + grad_j * m_i;
                                    ParticlePosition<DIM> r_ij = config.displacement(p_j, p_i);
                                    backward_smoothing_kernel_inplace(r_ij, config.eps, grad_coef, output);
                                }
                            }
                            for (int d = 0; d < DIM; ++d)
                                dL_dx_buffer[i * DIM + d] += output[d] * config.smoothing_coef;
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::Blur)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk * feature_dim; i += blockDim.x)
                        {
                            uint32_t particle_idx = features.row(i);
                            uint32_t feature_idx = features.col(i);
                            ParticlePosition<DIM> p_i = local_pos[particle_idx];
                            float grad_i, s_i, v_i;
                            float grad_j, s_j, v_j;
                            float w_ij;
                            if constexpr (!COMPUTE_DS_ONLY)
                            {
                                grad_i = dL_local[i];
                                s_i = local_state[i];
                            }
                            v_i = local_density_rcp[particle_idx];
                            float v_i_sq_over_m_i = local_mass[particle_idx];
                            float output_rho = 0.0f;
                            float output_dx[DIM] = {0.0f};
                            float output_s = 0.0f;
                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                ParticlePosition<DIM> p_j = neighbor_pos[j];
                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    ParticlePosition<DIM> r_ij = config.displacement(p_j, p_i);
                                    w_ij = smoothing_kernel(r_ij, config.eps);
                                    grad_j = dL_neighbor[j * feature_dim + feature_idx];
                                    output_s += grad_j * w_ij * v_i;
                                    if constexpr (!COMPUTE_DS_ONLY)
                                    {
                                        v_j = neighbor_density_rcp[j];
                                        s_j = neighbor_state[j * feature_dim + feature_idx];
                                        output_rho += -grad_j * s_i * w_ij * v_i_sq_over_m_i;
                                        backward_smoothing_kernel_inplace(r_ij, config.eps,
                                                                          grad_i * s_j * v_j + grad_j * s_i * v_i,
                                                                          output_dx);
                                    }
                                }
                            }
                            dL_ds_buffer[i] += output_s * config.smoothing_coef;
                            if constexpr (!COMPUTE_DS_ONLY)
                            {
                                atomicAdd(&dL_drho_buffer[particle_idx], output_rho * config.smoothing_coef);
                                for (int d = 0; d < DIM; ++d)
                                    atomicAdd(&dL_dx_buffer[particle_idx * DIM + d], output_dx[d] * config.smoothing_coef);
                            }
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::DensityGradient)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk; i += blockDim.x)
                        {
                            ParticlePosition<DIM> p_i = local_pos[i];
                            float grad_i[DIM];
                            for (int d = 0; d < DIM; ++d)
                                grad_i[d] = dL_local[i * DIM + d];
                            float m_i = local_mass[i];

                            float output[DIM] = {0.0f};
                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                ParticlePosition<DIM> p_j = neighbor_pos[j];

                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    float grad_diff[DIM];
                                    float m_j = neighbor_mass[j];
                                    for (int d = 0; d < DIM; ++d)
                                        grad_diff[d] = dL_neighbor[j * DIM + d] * m_i - grad_i[d] * m_j;
                                    ParticlePosition<DIM> r_ij = config.displacement(p_j, p_i);
                                    backward_spiky_kernel_inplace(r_ij, config.eps, grad_diff, output);
                                }
                            }
                            for (int d = 0; d < DIM; ++d)
                                dL_dx_buffer[i * DIM + d] += output[d] * config.spiky_coef;
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::Gradient)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk * feature_dim; i += blockDim.x)
                        {
                            uint32_t particle_idx = features.row(i);
                            uint32_t feature_idx = features.col(i);
                            ParticlePosition<DIM> p_i = local_pos[particle_idx];
                            float grad_i[DIM], s_i, v_i;
                            float grad_j[DIM], s_j, v_j;
                            float grad_diff[DIM];
                            if constexpr (!COMPUTE_DS_ONLY)
                            {
                                s_i = local_state[i];
                            }
                            for (int d = 0; d < DIM; ++d)
                                grad_i[d] = dL_local[i * DIM + d];
                            v_i = local_density_rcp[particle_idx];
                            float v_i_sq_over_m_i = local_mass[particle_idx];
                            float output_rho = 0.0f;
                            float output_dx[DIM] = {0.0f};
                            float output_s = 0.0f;
                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                ParticlePosition<DIM> p_j = neighbor_pos[j];
                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    float w_ij[DIM] = {0.0f};
                                    ParticlePosition<DIM> r_ij = config.displacement(p_j, p_i);
                                    spiky_kernel_inplace(r_ij, config.eps, 1.0f, w_ij);
                                    for (int d = 0; d < DIM; ++d)
                                        grad_j[d] = dL_neighbor[(j * feature_dim + feature_idx) * DIM + d];
                                    v_j = neighbor_density_rcp[j];
                                    if constexpr (!COMPUTE_DS_ONLY)
                                    {
                                        s_j = neighbor_state[j * feature_dim + feature_idx];
                                    }

                                    for (int d = 0; d < DIM; ++d)
                                    {
                                        output_s += (grad_j[d] * v_i + grad_i[d] * v_j) * w_ij[d];
                                        if constexpr (!COMPUTE_DS_ONLY)
                                        {
                                            output_rho += -grad_j[d] * (s_i - s_j) * w_ij[d] * v_i_sq_over_m_i;
                                            grad_diff[d] = (s_i - s_j) * (grad_i[d] * v_j + grad_j[d] * v_i);
                                        }
                                    }
                                    if constexpr (!COMPUTE_DS_ONLY)
                                    {
                                        backward_spiky_kernel_inplace(r_ij, config.eps, grad_diff, output_dx);
                                    }
                                }
                            }
                            dL_ds_buffer[i] += output_s * config.spiky_coef;
                            if constexpr (!COMPUTE_DS_ONLY)
                            {
                                atomicAdd(&dL_drho_buffer[particle_idx], output_rho * config.spiky_coef);
                                for (int d = 0; d < DIM; ++d)
                                    atomicAdd(&dL_dx_buffer[particle_idx * DIM + d], output_dx[d] * config.spiky_coef);
                            }
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk; i += blockDim.x)
                        {
                            ParticlePosition<DIM> p_i = local_pos[i];
                            float grad_i[DIM * DIM], grad_j[DIM * DIM];
                            float v_i, v_j;
                            for (int idx = 0; idx < DIM * DIM; ++idx)
                                grad_i[idx] = dL_local[i * DIM * DIM + idx];
                            v_i = local_density_rcp[i];
                            float v_i_sq_over_m_i = local_mass[i];
                            float output_rho = 0.0f;
                            float output_dx[DIM] = {0.0f};
                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                ParticlePosition<DIM> p_j = neighbor_pos[j];

                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    ParticlePosition<DIM> r_ij = config.displacement(p_j, p_i);
                                    float r[DIM];
                                    for (int d = 0; d < DIM; ++d)
                                        r[d] = r_ij[d];
                                    v_j = neighbor_density_rcp[j];
                                    float w_ij[DIM] = {0.0f};
                                    float grad_diff[DIM * DIM];

                                    spiky_kernel_inplace(r_ij, config.eps, 1.0f, w_ij);

                                    for (uint32_t d1 = 0; d1 < DIM; ++d1)
                                        for (uint32_t d2 = 0; d2 < DIM; ++d2)
                                        {
                                            grad_j[d1 * DIM + d2] = dL_neighbor[j * DIM * DIM + d1 * DIM + d2];
                                            output_rho += -grad_j[d1 * DIM + d2] * r[d1] * w_ij[d2] * v_i_sq_over_m_i;
                                            float c = grad_i[d1 * DIM + d2] * v_j + grad_j[d1 * DIM + d2] * v_i;
                                            output_dx[d1] += w_ij[d2] * c;
                                            grad_diff[d1 * DIM + d2] = r[d1] * c;
                                        }

                                    for (uint32_t d = 0; d < DIM; ++d)
                                        backward_spiky_kernel_inplace(r_ij, config.eps, &grad_diff[d * DIM], output_dx);
                                }
                            }
                            dL_drho_buffer[i] += output_rho * config.spiky_coef;
                            for (int d = 0; d < DIM; ++d)
                                dL_dx_buffer[i * DIM + d] += output_dx[d] * config.spiky_coef;
                        }
                    }

                    __syncthreads();
                }
            }
        }

        // Write back accumulated gradients for local particles
        if constexpr (!COMPUTE_DS_ONLY)
        {
            for (uint32_t k = threadIdx.x; k < local_chunk; k += blockDim.x)
            {
                if constexpr (DIM == 2)
                {
                    dL_dx_global[k].pos = make_float2(dL_dx_buffer[k * DIM + 0],
                                                      dL_dx_buffer[k * DIM + 1]);
                }
                else if constexpr (DIM == 3)
                {
                    dL_dx_global[k].pos = make_float3(dL_dx_buffer[k * DIM + 0],
                                                      dL_dx_buffer[k * DIM + 1],
                                                      dL_dx_buffer[k * DIM + 2]);
                }
            }
        }

        if constexpr (!COMPUTE_DS_ONLY)
        {
            if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::MomentMatrix)
            {
                for (uint32_t k = threadIdx.x; k < local_chunk; k += blockDim.x)
                    dL_drho_global[k] = dL_drho_buffer[k];
            }
        }

        if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient)
        {
            for (uint32_t k = threadIdx.x; k < local_chunk * feature_dim; k += blockDim.x)
                dL_ds_global[k] = dL_ds_buffer[k];
        }
        // Vector position writes read components initialized by other threads.
        // Finish those reads before a later local chunk clears the shared buffers.
        __syncthreads();
    }
}

template <int DIM, ComputationType COMP_TYPE, int STRIDE = 128, BoundaryCondition BC = BoundaryCondition::PERIODIC>
__global__ void forward_kernel(
    const ParticlePosition<DIM> * __restrict__ position_in, // Shape (batch_size, N, DIM)
    const float * __restrict__ rho_in,                      // Input density tensor, Shape (batch_size, N)
    const float * __restrict__ mass_in,                     // Input mass tensor, Shape (batch_size, N)
    const float * __restrict__ state_in,                    // Input state tensor, Shape (batch_size, N, feature_dim)
    float * __restrict__ rho_out,                           // Output/Input density tensor, Shape (batch_size, N)
    float * __restrict__ count_out,                         // Output count tensor, Shape (batch_size, N)
    float * __restrict__ blur_state_out,                    // Output state tensor, Shape (batch_size, N, feature_dim)
    float * __restrict__ grad_state_out,                    // Output state gradient tensor, Shape (batch_size, N, feature_dim, DIM)
    float * __restrict__ grad_density_out,                  // Output density gradient tensor, Shape (batch_size, N, DIM)
    float * __restrict__ moment_matrix_out,                 // Output moment matrix tensor, Shape (batch_size, N, DIM, DIM)
    uint32_t feature_dim,                     // Number of features in state
    const uint32_t * __restrict__ bin_offsets, // Shape (batch_size * cell_count + 1)
    const BlockInfo * __restrict__ block_info, // Block information array
    const GridConfig<DIM, BC> config)
{
    BlockInfo b_info = block_info[blockIdx.x];
    if (b_info.particle_count == 0)
        return;

    __shared__ ParticlePosition<DIM> local_pos[STRIDE];
    __shared__ ParticlePosition<DIM> neighbor_pos[STRIDE];
    extern __shared__ float smem[];
    float *local_buffer = smem;
    float *neighbor_state, *local_state, *neighbor_density_rcp, *neighbor_mass, *output_buffer;
    uint32_t output_dim = 1;
    const RowSplit features(feature_dim);
    if constexpr (COMP_TYPE == ComputationType::Blur)
    {
        neighbor_state = &local_buffer[feature_dim * STRIDE];
        neighbor_density_rcp = &neighbor_state[feature_dim * STRIDE];
        neighbor_mass = &neighbor_density_rcp[STRIDE];
        output_buffer = blur_state_out;
        output_dim = feature_dim;
    }
    else if constexpr (COMP_TYPE == ComputationType::Gradient)
    {
        neighbor_state = &local_buffer[feature_dim * DIM * STRIDE];
        local_state = &neighbor_state[feature_dim * STRIDE];
        neighbor_density_rcp = &local_state[feature_dim * STRIDE];
        neighbor_mass = &neighbor_density_rcp[STRIDE];
        output_buffer = grad_state_out;
        output_dim = feature_dim * DIM;
    }
    else if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
    {
        neighbor_density_rcp = &local_buffer[DIM * DIM * STRIDE];
        neighbor_mass = &neighbor_density_rcp[STRIDE];
        output_buffer = moment_matrix_out;
        output_dim = DIM * DIM;
    }
    else if constexpr (COMP_TYPE == ComputationType::DensityGradient)
    {
        neighbor_mass = &local_buffer[DIM * STRIDE];
        output_buffer = grad_density_out;
        output_dim = DIM;
    }
    else if constexpr (COMP_TYPE == ComputationType::Density)
    {
        neighbor_mass = &local_buffer[STRIDE];
        output_buffer = rho_out;
    }
    else
    {
        output_buffer = count_out;
        neighbor_mass = nullptr;
    }

    uint32_t batch_idx = b_info.batch_idx;
    int cell[DIM];
    config.hash2cell(b_info.cell_idx % config.cell_count, cell);

    // Batch-local indices of this grid block's particles.
    uint32_t local_start = bin_offsets[b_info.cell_idx] - batch_idx * config.particle_count + b_info.offset_in_cell;
    uint32_t local_end = bin_offsets[b_info.cell_idx + 1] - batch_idx * config.particle_count;
    local_end = min(local_end, local_start + b_info.particle_count);
    uint32_t local_count = local_end - local_start;
    float eps2 = config.eps * config.eps;
    float default_mass = config.default_mass;

    // Local particles in chunks of STRIDE; each chunk visits all neighbor tiles.
    for (uint32_t local_offset = 0; local_offset < local_count; local_offset += STRIDE)
    {
        uint32_t local_chunk = min(STRIDE, local_count - local_offset);

        for (uint32_t k = threadIdx.x; k < local_chunk; k += blockDim.x)
            local_pos[k] = read_position<DIM>(position_in, config.particle_count, batch_idx,
                                              local_start + local_offset + k);
        if constexpr (COMP_TYPE == ComputationType::Gradient)
        {
            for (uint32_t k = threadIdx.x; k < local_chunk * feature_dim; k += blockDim.x)
                local_state[k] = read_state(state_in, config.particle_count, feature_dim, batch_idx,
                                            local_start + local_offset + features.row(k), features.col(k));
        }

        float *output_buffer_global = output_buffer + (batch_idx * config.particle_count + local_start + local_offset) * output_dim;
        for (uint32_t k = threadIdx.x; k < local_chunk * output_dim; k += blockDim.x)
            local_buffer[k] = 0.0f;

        // Same spans as GridConfig::neighbor_span, computed inline: in these
        // block-wide kernels the inline form compiles to faster code.
        for (int row = 0; row < GridConfig<DIM, BC>::NEIGHBOR_ROWS; ++row)
        {
            int neighbor[DIM];
            if (!config.neighbor_row(cell, row, neighbor))
                continue;
            for (int wrap = 0; wrap < GridConfig<DIM, BC>::WRAP_CHUNKS; ++wrap)
            {
                int first_x, last_x;
                if (!config.row_x_range(cell, wrap, first_x, last_x))
                    continue;
                neighbor[0] = first_x;
                uint32_t first = config.cell2hash(neighbor);
                neighbor[0] = last_x;
                uint32_t last = config.cell2hash(neighbor);
                // Batch-local indices, as read_position and read_state expect.
                uint32_t remote_start = bin_offsets[batch_idx * config.cell_count + first] - batch_idx * config.particle_count;
                uint32_t remote_end = bin_offsets[batch_idx * config.cell_count + last + 1] - batch_idx * config.particle_count;
                uint32_t remote_count = remote_end - remote_start;
                if (remote_count == 0)
                    continue;

                for (uint32_t remote_offset = 0; remote_offset < remote_count; remote_offset += STRIDE)
                {
                    uint32_t remote_chunk = min(STRIDE, remote_count - remote_offset);
                    // Stage this tile of neighbors (positions, then what the operator needs).
                    for (uint32_t k = threadIdx.x; k < remote_chunk; k += blockDim.x)
                        neighbor_pos[k] = read_position<DIM>(position_in,
                                                             config.particle_count, batch_idx,
                                                             remote_start + remote_offset + k);
                    if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::MomentMatrix)
                    {
                        for (uint32_t k = threadIdx.x; k < remote_chunk; k += blockDim.x)
                        {
                            uint32_t idx = remote_start + remote_offset + k;
                            neighbor_mass[k] = read_mass(mass_in, config.particle_count, batch_idx, idx, default_mass);
                            float rho_j = read_state(rho_in, config.particle_count, 1, batch_idx, idx, 0);
                            neighbor_density_rcp[k] = neighbor_mass[k] / rho_j;
                        }
                    }
                    else if constexpr (COMP_TYPE == ComputationType::Density || COMP_TYPE == ComputationType::DensityGradient)
                    {
                        for (uint32_t k = threadIdx.x; k < remote_chunk; k += blockDim.x)
                        {
                            uint32_t idx = remote_start + remote_offset + k;
                            neighbor_mass[k] = read_mass(mass_in, config.particle_count, batch_idx, idx, default_mass);
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient)
                    {
                        for (uint32_t k = threadIdx.x; k < remote_chunk * feature_dim; k += blockDim.x)
                            neighbor_state[k] = read_state(state_in, config.particle_count, feature_dim, batch_idx,
                                                           remote_start + remote_offset + features.row(k), features.col(k));
                    }
                    __syncthreads();

                    // Local/neighbor interactions: one thread per local particle,
                    // or per (particle, feature) for blur and gradient.
                    if constexpr (COMP_TYPE == ComputationType::Count || COMP_TYPE == ComputationType::Density)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk; i += blockDim.x)
                        {
                            ParticlePosition<DIM> p_i = local_pos[i];
                            float output = 0.0f;
                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                ParticlePosition<DIM> p_j = neighbor_pos[j];
                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    if constexpr (COMP_TYPE == ComputationType::Density)
                                    {
                                        ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                                        output += smoothing_kernel(r_ij, config.eps) * neighbor_mass[j];
                                    }
                                    if constexpr (COMP_TYPE == ComputationType::Count)
                                        output += 1.0f;
                                }
                            }
                            if constexpr (COMP_TYPE == ComputationType::Density)
                                local_buffer[i] += output * config.smoothing_coef;
                            if constexpr (COMP_TYPE == ComputationType::Count)
                                local_buffer[i] += output;
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::Blur)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk * output_dim; i += blockDim.x)
                        {
                            uint32_t particle_idx = features.row(i);
                            uint32_t feature_idx = features.col(i);
                            ParticlePosition<DIM> p_i = local_pos[particle_idx];
                            float output = 0.0f;
                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                ParticlePosition<DIM> p_j = neighbor_pos[j];
                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    float v_j = neighbor_density_rcp[j];
                                    float s_j = neighbor_state[j * feature_dim + feature_idx];
                                    ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                                    float w_ij = smoothing_kernel(r_ij, config.eps);
                                    output += s_j * w_ij * v_j;
                                }
                            }
                            local_buffer[i] += output * config.smoothing_coef;
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::Gradient)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk * feature_dim; i += blockDim.x)
                        {
                            uint32_t particle_idx = features.row(i);
                            uint32_t feature_idx = features.col(i);
                            float s_i = local_state[i];
                            ParticlePosition<DIM> p_i = local_pos[particle_idx];
                            float output[DIM] = {0.0f};
                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                ParticlePosition<DIM> p_j = neighbor_pos[j];
                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    float v_j = neighbor_density_rcp[j];
                                    float s_j = neighbor_state[j * feature_dim + feature_idx];
                                    ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                                    spiky_kernel_inplace(r_ij, config.eps, (s_j - s_i) * v_j, output);
                                }
                            }
                            for (uint32_t dim = 0; dim < DIM; ++dim)
                                local_buffer[i * DIM + dim] += output[dim] * config.spiky_coef;
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::DensityGradient)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk; i += blockDim.x)
                        {
                            ParticlePosition<DIM> p_i = local_pos[i];
                            float output[DIM] = {0.0f};
                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                ParticlePosition<DIM> p_j = neighbor_pos[j];
                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                                    spiky_kernel_inplace(r_ij, config.eps, neighbor_mass[j], output);
                                }
                            }
                            for (uint32_t dim = 0; dim < DIM; ++dim)
                                local_buffer[i * DIM + dim] += output[dim] * config.spiky_coef;
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk; i += blockDim.x)
                        {
                            ParticlePosition<DIM> p_i = local_pos[i];
                            float output[DIM * DIM] = {0.0f};
                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                ParticlePosition<DIM> p_j = neighbor_pos[j];
                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    float v_j = neighbor_density_rcp[j];
                                    ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                                    moment_matrix_inplace(r_ij, config.eps, v_j, output);
                                }
                            }
                            for (uint32_t m = 0; m < DIM * DIM; ++m)
                                local_buffer[i * DIM * DIM + m] += output[m] * config.spiky_coef;
                        }
                    }
                    __syncthreads();
                }
            }
        }

        for (uint32_t k = threadIdx.x; k < local_chunk * output_dim; k += blockDim.x)
            output_buffer_global[k] = local_buffer[k];
    }
}

template <int DIM, BoundaryCondition BC>
class HashGrid
{
    static_assert(DIM == 2 || DIM == 3, "HashGrid only supports 2D or 3D");

protected:
    GridConfig<DIM, BC> m_config;

    // One counter allocation, cleared by a single memset: cell counts (with a
    // zero scan sentinel), then blocks per cell.
    uint32_t *m_cell_counts = nullptr;       // Device array of size (cell_count * batch_size + 1)
    size_t m_counter_words = 0;

    // Pre-allocated temporary storage for CUB prefix sum
    void *m_temp_storage = nullptr;
    size_t m_temp_storage_bytes = 0;

    uint32_t *m_blocks_per_cell = nullptr; // Part of m_cell_counts: blocks needed per cell (batch_size * cell_count + 1)
    uint32_t *m_block_offsets = nullptr;   // Device array for cumulative sum of blocks per cell (batch_size * cell_count + 1)
    float *m_unit_mass = nullptr;          // batch_size * N ones for fused perception without masses

    uint32_t m_dimension;  // Floats per position: DIM
    uint32_t m_max_blocks; // Number of blocks used in the last computation

    void _init(const int (&grid_size)[DIM], float eps, uint32_t particle_count, uint32_t batch_size,
               uint32_t max_particles_per_block)
    {
        m_config.eps = eps;
        m_config.inv_eps = 1.0f / eps;
        m_config.particle_count = particle_count;
        m_config.batch_size = batch_size;
        m_config.MAX_PARTICLES_PER_BLOCK = max_particles_per_block;
        m_config.default_mass = 1.0f;
        m_config.cell_count = 1;
        m_config.is_pow2 = true;
        for (int d = 0; d < DIM; ++d)
        {
            m_config.grid_size[d] = grid_size[d];
            m_config.cell_count *= uint32_t(grid_size[d]);
            m_config.is_pow2 = m_config.is_pow2 && (grid_size[d] & (grid_size[d] - 1)) == 0;
            m_config.L[d] = static_cast<float>(grid_size[d]) * eps;
            m_config.half_L[d] = 0.5f * m_config.L[d];
            m_config.inv_L[d] = 1.0f / m_config.L[d];
        }
        m_config.max_blocks_per_batch = min(particle_count,
                                            m_config.cell_count + (particle_count + m_config.MAX_PARTICLES_PER_BLOCK - 1) / m_config.MAX_PARTICLES_PER_BLOCK);
        m_max_blocks = m_config.max_blocks_per_batch * batch_size;

        float inv_eps2 = m_config.inv_eps * m_config.inv_eps;
        float inv_eps4 = inv_eps2 * inv_eps2;

        if constexpr (DIM == 2)
        {
            float inv_eps5 = inv_eps4 * m_config.inv_eps;
            float inv_eps8 = inv_eps4 * inv_eps4;
            m_config.smoothing_coef = 4.0f / M_PI * inv_eps8;
            m_config.spiky_coef = 10.0f / M_PI * inv_eps5;
        }
        else if constexpr (DIM == 3)
        {
            float inv_eps6 = inv_eps4 * inv_eps2;
            float inv_eps9 = inv_eps4 * inv_eps4 * m_config.inv_eps;
            m_config.smoothing_coef = (315.0f / (64.0f * M_PI)) * inv_eps9;
            m_config.spiky_coef = (15.0f / M_PI) * inv_eps6;
        }

        // Allocate cell counts array
        // Exclusive scans include a zero sentinel to produce the final total.
        size_t scan_words = m_config.cell_count * size_t(batch_size) + 1;
        m_counter_words = 2 * scan_words;
        cuda_check(cudaMalloc(&m_cell_counts, sizeof(uint32_t) * m_counter_words));

        m_blocks_per_cell = m_cell_counts + scan_words;

        // Temporary storage of the zipped prefix sum over (cell count, blocks per cell).
        cub::DeviceScan::ExclusiveScan(
            m_temp_storage, m_temp_storage_bytes,
            thrust::make_zip_iterator(thrust::make_tuple(m_cell_counts, m_blocks_per_cell)),
            thrust::make_zip_iterator(thrust::make_tuple((uint32_t *)nullptr, (uint32_t *)nullptr)),
            CounterPairSum{}, thrust::make_tuple(0u, 0u), scan_words);

        cuda_check(cudaMalloc(&m_temp_storage, m_temp_storage_bytes));
        cuda_check(cudaMalloc(&m_block_offsets, sizeof(uint32_t) * scan_words));
        // Omitted masses are ones; a persistent buffer avoids filling and
        // binning one every step (binned ones are ones).
        std::vector<float> ones(size_t(particle_count) * batch_size, 1.0f);
        cuda_check(cudaMalloc(&m_unit_mass, sizeof(float) * ones.size()));
        cuda_check(cudaMemcpy(m_unit_mass, ones.data(), sizeof(float) * ones.size(), cudaMemcpyHostToDevice));

        m_dimension = DIM;
    }

public:
    ~HashGrid()
    {
        cudaFree(m_cell_counts);  // Also holds m_blocks_per_cell.
        cudaFree(m_temp_storage);
        cudaFree(m_block_offsets);
        cudaFree(m_unit_mass);
    }

    float eps() const { return m_config.eps; }
    uint32_t particle_count() const { return m_config.particle_count; }
    uint32_t batch_size() const { return m_config.batch_size; }

    int grid_size_x() const { return m_config.grid_size[0]; }
    int grid_size_y() const { return m_config.grid_size[1]; }

    int max_blocks() const { return m_max_blocks; }

    uint32_t max_particles_per_block() const { return m_config.MAX_PARTICLES_PER_BLOCK; }

    // Particles to bin together with the scatter pass (fused path).
    struct Binning
    {
        const float *state;
        const float *mass;
        ParticlePosition<DIM> *binned_positions;
        float *binned_state;
        float *binned_mass;
        uint32_t features;
    };

    // Counting sort of particles into cells plus grid-block assignment, ordered
    // on `stream`. Every output entry is written (assign_block_info also writes
    // the sentinel tail), so only the counters need clearing. With `binning`,
    // the scatter pass also copies positions, states and masses into binned order.
    void initialize_impl(const ParticlePosition<DIM> *d_positions, uint32_t *d_permutation,
                         uint32_t *d_bin_offsets, BlockInfo *d_block_info, cudaStream_t stream,
                         const Binning *binning = nullptr)
    {
        uint32_t scan_items = m_config.batch_size * m_config.cell_count + 1;
        cuda_check(cudaMemsetAsync(m_cell_counts, 0, sizeof(uint32_t) * m_counter_words, stream));

        dim3 block_dim(256, 1, 1);
        dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
        grid_count_particles<<<grid_dim, block_dim, 0, stream>>>(
            d_positions, m_cell_counts, m_blocks_per_cell, d_permutation, m_config);
        cuda_check(cudaGetLastError());

        // Both prefix sums at once: bin offsets (where each cell starts) and
        // block offsets (the first grid block of each cell).
        cuda_check(cub::DeviceScan::ExclusiveScan(
            m_temp_storage, m_temp_storage_bytes,
            thrust::make_zip_iterator(thrust::make_tuple(m_cell_counts, m_blocks_per_cell)),
            thrust::make_zip_iterator(thrust::make_tuple(d_bin_offsets, m_block_offsets)),
            CounterPairSum{}, thrust::make_tuple(0u, 0u), scan_items, stream));

        // Scatter: binned index of each particle (and the binned copy, fused path)
        if (binning)
            grid_scatter_and_bin<<<grid_dim, block_dim, 0, stream>>>(
                d_positions, binning->state, binning->mass, d_bin_offsets, d_permutation,
                binning->binned_positions, binning->binned_state, binning->binned_mass,
                binning->features, m_config);
        else
            grid_compute_permutation<<<grid_dim, block_dim, 0, stream>>>(
                d_positions, d_bin_offsets, d_permutation, m_config);
        cuda_check(cudaGetLastError());

        // Launch kernel to assign block information (it also clears the unused tail)
        block_dim = dim3(256, 1, 1);
        grid_dim = dim3((m_config.cell_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
        assign_block_info<<<grid_dim, block_dim, 0, stream>>>(
            m_cell_counts, m_block_offsets, d_block_info, m_config);
        cuda_check(cudaGetLastError());
    }

    void initialize(
        FloatArray3D positions,   // Shape (batch_size, N, DIM)
        IntArray2D permutation,   // Shape (batch_size, N)
        IntArray1D bin_offsets,   // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info // Shape (max_blocks, 4)
    )
    {
        validate_array_shape(positions, per_particle({m_dimension}), "positions");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(permutation, per_particle(), "permutation");
        validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        initialize_impl(reinterpret_cast<const ParticlePosition<DIM> *>(positions.data()),
                        reinterpret_cast<uint32_t *>(permutation.data()),
                        reinterpret_cast<uint32_t *>(bin_offsets.data()),
                        reinterpret_cast<BlockInfo *>(block_info.data()), 0);
    }

    void bin_particles(
        FloatArray3D input_positions,  // Shape (batch_size, N, DIM)
        FloatArray3D input_state,      // Shape (batch_size, N, feature_dim)
        FloatArray2D input_mass,       // Shape (batch_size, N)
        IntArray2D permutation,        // Shape (batch_size, N)
        FloatArray3D output_positions, // Shape (batch_size, N, DIM)
        FloatArray3D output_state,     // Shape (batch_size, N, feature_dim)
        FloatArray2D output_mass       // Shape (batch_size, N)
    )
    {
        uint32_t feature_dim = input_state.shape(2);
        validate_array_shape(input_positions, per_particle({m_dimension}), "input_positions");
        validate_array_shape(input_state, per_particle({feature_dim}), "input_state");
        validate_array_shape(input_mass, per_particle(), "input_mass");
        validate_array_shape(permutation, per_particle(), "permutation");
        validate_array_shape(output_positions, per_particle({m_dimension}), "output_positions");
        validate_array_shape(output_state, per_particle({feature_dim}), "output_state");
        validate_array_shape(output_mass, per_particle(), "output_mass");

        // Cast device pointers
        const ParticlePosition<DIM> *d_input_positions = reinterpret_cast<const ParticlePosition<DIM> *>(input_positions.data());
        ParticlePosition<DIM> *d_output_positions = reinterpret_cast<ParticlePosition<DIM> *>(output_positions.data());
        const uint32_t *d_permutation = reinterpret_cast<const uint32_t *>(permutation.data());

        dim3 block_dim(256, 1, 1);
        dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
        apply_permutation<<<grid_dim, block_dim>>>(
            d_input_positions, input_state.data(), input_mass.data(), d_permutation,
            d_output_positions, output_state.data(), output_mass.data(),
            feature_dim, m_config);
    }

    void backward_bin_particles(
        FloatArray3D grad_output_positions, // Shape (batch_size, N, DIM)
        FloatArray3D grad_output_state,     // Shape (batch_size, N, feature_dim)
        FloatArray2D grad_output_mass,      // Shape (batch_size, N)
        IntArray2D permutation,             // Shape (batch_size, N)
        FloatArray3D grad_input_positions,  // Shape (batch_size, N, DIM)
        FloatArray3D grad_input_state,      // Shape (batch_size, N, feature_dim)
        FloatArray2D grad_input_mass        // Shape (batch_size, N)
    )
    {
        uint32_t feature_dim = grad_output_state.shape(2);
        validate_array_shape(grad_output_positions, per_particle({m_dimension}), "grad_output_positions");
        validate_array_shape(grad_output_state, per_particle({feature_dim}), "grad_output_state");
        validate_array_shape(grad_output_mass, per_particle(), "grad_output_mass");
        validate_array_shape(permutation, per_particle(), "permutation");
        validate_array_shape(grad_input_positions, per_particle({m_dimension}), "grad_input_positions");
        validate_array_shape(grad_input_state, per_particle({feature_dim}), "grad_input_state");
        validate_array_shape(grad_input_mass, per_particle(), "grad_input_mass");

        // Cast device pointers
        const ParticlePosition<DIM> *d_grad_output_positions = reinterpret_cast<const ParticlePosition<DIM> *>(grad_output_positions.data());
        ParticlePosition<DIM> *d_grad_input_positions = reinterpret_cast<ParticlePosition<DIM> *>(grad_input_positions.data());
        const uint32_t *d_permutation = reinterpret_cast<const uint32_t *>(permutation.data());

        dim3 block_dim(256, 1, 1);
        dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
        backward_apply_permutation<<<grid_dim, block_dim>>>(
            d_grad_output_positions, grad_output_state.data(), grad_output_mass.data(), d_permutation,
            d_grad_input_positions, grad_input_state.data(), grad_input_mass.data(),
            feature_dim, m_config);
    }

    // Fused NPA perception (fused_perception.cuh, FUSED_PERCEPTION.md): fixed masses,
    // detached positions, native channel counts 4/8/16/32 (sph.py pads others).
    // One forward call builds the grid, bins, and perceives; one backward call
    // propagates the perception adjoint, adds the binned-state cotangent, and
    // unbins. `snapshot` holds int32 words [block_info (max_blocks*4) |
    // bin_offsets (B*cells+1) | permutation (B*N)]; `workspace` holds the
    // forward state saved for backward, laid out by fused_workspace_layout.
    struct FusedWorkspace {
        size_t volume, density_gradient, gradient_sum, gradient_values, mass, total_mass, size;
    };

    FusedWorkspace fused_workspace_layout(uint32_t features, bool density_grad, bool normalize_state_grad,
                                          bool save_backward, bool has_mass) const
    {
        size_t particles = size_t(m_config.batch_size) * m_config.particle_count;
        FusedWorkspace layout{};
        layout.volume = 0;
        layout.density_gradient = particles;
        layout.gradient_sum = layout.density_gradient + (density_grad ? particles * DIM : 0);
        layout.gradient_values = layout.gradient_sum + (save_backward ? particles * DIM : 0);
        layout.mass = layout.gradient_values + (save_backward && normalize_state_grad ? particles * features * DIM : 0);
        layout.total_mass = layout.mass + (has_mass ? particles : 0);
        layout.size = layout.total_mass + (has_mass && density_grad ? m_config.batch_size : 0);
        return layout;
    }

    size_t fused_workspace_size(uint32_t features, bool density_grad, bool normalize_state_grad,
                                bool save_backward, bool has_mass) const
    {
        return fused_workspace_layout(features, density_grad, normalize_state_grad, save_backward, has_mass).size;
    }

    size_t fused_snapshot_words() const
    {
        return size_t(m_max_blocks) * 4 + size_t(m_config.batch_size) * m_config.cell_count + 1
             + size_t(m_config.batch_size) * m_config.particle_count;
    }

    // The snapshot's int32 words: [block_info (max_blocks*4) | bin_offsets (B*cells+1) | permutation (B*N)].
    struct SnapshotView
    {
        BlockInfo *block_info;
        uint32_t *bin_offsets;
        uint32_t *permutation;
    };

    SnapshotView snapshot_view(IntArray1D &snapshot) const
    {
        auto *words = reinterpret_cast<uint32_t *>(snapshot.data());
        uint32_t *bin_offsets = words + size_t(m_max_blocks) * 4;
        return {reinterpret_cast<BlockInfo *>(words), bin_offsets,
                bin_offsets + size_t(m_config.batch_size) * m_config.cell_count + 1};
    }

    void fused_perception_forward(
        FloatArray3D positions, FloatArray3D state, std::optional<FloatArray2D> mass, FloatArray1D eps0,
        IntArray1D snapshot, FloatArray3D binned_positions, FloatArray3D binned_state,
        FloatArray3D perception, FloatArray1D workspace,
        bool perceive_density_grad, bool normalize_state_grad, bool normalize_density_grad,
        bool save_backward, uint64_t stream = 0)
    {
        uint32_t features = state.shape(2);
        uint32_t output_dim = features * (2 + DIM) + (perceive_density_grad ? DIM : 0);
        FusedWorkspace layout = fused_workspace_layout(
            features, perceive_density_grad, normalize_state_grad, save_backward, mass.has_value());
        validate_array_shape(positions, per_particle({m_dimension}), "positions");
        validate_array_shape(state, per_particle({features}), "state");
        if (mass)
            validate_array_shape(*mass, per_particle(), "mass");
        validate_array_shape(eps0, {1}, "eps0");
        validate_array_shape(snapshot, {fused_snapshot_words()}, "snapshot");
        validate_array_shape(binned_positions, per_particle({m_dimension}), "binned_positions");
        validate_array_shape(binned_state, per_particle({features}), "binned_state");
        validate_array_shape(perception, per_particle({output_dim}), "perception");
        validate_array_shape(workspace, {layout.size}, "workspace");
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        auto [block_info, bin_offsets, permutation] = snapshot_view(snapshot);
        auto *x = reinterpret_cast<const ParticlePosition<DIM> *>(positions.data());
        auto *x_bin = reinterpret_cast<ParticlePosition<DIM> *>(binned_positions.data());
        float *base = workspace.data();
        // Omitted masses read the persistent ones buffer (binned ones are ones).
        float *mass_bin = mass ? base + layout.mass : m_unit_mass;

        Binning binning{state.data(), mass ? mass->data() : nullptr, x_bin, binned_state.data(),
                        mass ? mass_bin : nullptr, features};
        initialize_impl(x, permutation, bin_offsets, block_info, cuda_stream, &binning);
        // Explicit masses: per-sample totals normalize the density gradient.
        const float *total_mass = nullptr;
        if (mass && perceive_density_grad) {
            fused_perception::total_mass<<<m_config.batch_size, fused_perception::TOTAL_MASS_THREADS, 0, cuda_stream>>>(
                mass->data(), m_config.particle_count, base + layout.total_mass);
            cuda_check(cudaGetLastError());
            total_mass = base + layout.total_mass;
        }
        launch_fused_density(x_bin, mass_bin, base + layout.volume,
                             perceive_density_grad ? base + layout.density_gradient : nullptr,
                             bin_offsets, block_info, perceive_density_grad, cuda_stream);
        launch_fused_features(x_bin, binned_state.data(), base + layout.volume,
                              base + layout.density_gradient, total_mass, eps0.data(), perception.data(),
                              base + layout.gradient_values, base + layout.gradient_sum,
                              bin_offsets, block_info, features, perceive_density_grad,
                              normalize_state_grad, normalize_density_grad, save_backward, cuda_stream);
    }

    void fused_perception_backward(
        FloatArray3D binned_positions, FloatArray1D workspace, std::optional<FloatArray3D> grad_perception,
        std::optional<FloatArray3D> grad_binned_positions, std::optional<FloatArray3D> grad_binned_state,
        FloatArray1D eps0, IntArray1D snapshot, FloatArray1D scratch,
        std::optional<FloatArray3D> grad_positions, std::optional<FloatArray3D> grad_state, uint32_t features,
        bool perceive_density_grad, bool normalize_state_grad, bool has_mass, uint64_t stream = 0)
    {
        uint32_t output_dim = features * (2 + DIM) + (perceive_density_grad ? DIM : 0);
        size_t particles = size_t(m_config.batch_size) * m_config.particle_count;
        // The saved forward state is read only for a perception cotangent, in
        // which case the forward ran with save_backward.
        FusedWorkspace layout = fused_workspace_layout(
            features, perceive_density_grad, normalize_state_grad, true, has_mass);
        validate_array_shape(binned_positions, per_particle({m_dimension}), "binned_positions");
        if (grad_perception)
            validate_array_shape(workspace, {layout.size}, "workspace");
        validate_array_shape(eps0, {1}, "eps0");
        validate_array_shape(snapshot, {fused_snapshot_words()}, "snapshot");
        if (grad_state)
            validate_array_shape(*grad_state, per_particle({features}), "grad_state");
        if (grad_perception) {
            validate_array_shape(*grad_perception, per_particle({output_dim}), "grad_perception");
            validate_array_shape(scratch, {particles * features * (DIM + 2)}, "scratch");
        }
        if (grad_binned_state)
            validate_array_shape(*grad_binned_state, per_particle({features}), "grad_binned_state");
        if (grad_state.has_value() != (grad_perception.has_value() || grad_binned_state.has_value()))
            throw std::runtime_error("grad_state requires a perception or binned-state cotangent, and vice versa");
        if (grad_positions.has_value() != grad_binned_positions.has_value())
            throw std::runtime_error("grad_positions and grad_binned_positions must be given together");
        if (grad_positions) {
            validate_array_shape(*grad_positions, per_particle({m_dimension}), "grad_positions");
            validate_array_shape(*grad_binned_positions, per_particle({m_dimension}), "grad_binned_positions");
        }
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        auto [block_info, bin_offsets, permutation] = snapshot_view(snapshot);
        const float *base = workspace.data();
        // Binned perception adjoint first; the binned-state cotangent is added
        // while unbinning (the sum autograd would otherwise form).
        const float *state_source = grad_binned_state ? grad_binned_state->data() : nullptr;
        const float *state_addend = nullptr;
        if (grad_perception) {
            // scratch: [incoming cotangents (B*N*(DIM+1)*C) | binned state adjoint (B*N*C)]
            float *perception_grad_state = scratch.data() + particles * features * (DIM + 1);
            launch_fused_backward(
                reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data()), base + layout.volume,
                grad_perception->data(), base + layout.gradient_values, base + layout.gradient_sum, eps0.data(),
                scratch.data(), perception_grad_state, bin_offsets, block_info, features,
                perceive_density_grad, normalize_state_grad, cuda_stream);
            state_addend = state_source;
            state_source = perception_grad_state;
        }
        dim3 block_dim(256, 1, 1);
        dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
        backward_apply_permutation<<<grid_dim, block_dim, 0, cuda_stream>>>(
            grad_binned_positions ? reinterpret_cast<const ParticlePosition<DIM> *>(grad_binned_positions->data()) : nullptr,
            state_source, nullptr, permutation,
            grad_positions ? reinterpret_cast<ParticlePosition<DIM> *>(grad_positions->data()) : nullptr,
            grad_state ? grad_state->data() : nullptr, nullptr, features, m_config, state_addend);
        cuda_check(cudaGetLastError());
    }

private:
    // Grid size of a persistent fused kernel: eight waves of resident CUDA blocks
    // (occupancy queried once per kernel), never more than the grid blocks.
    template <typename Kernel>
    unsigned int persistent_grid(Kernel kernel, int threads) const
    {
        static std::map<const void *, unsigned int> cache;
        auto key = reinterpret_cast<const void *>(kernel);
        auto found = cache.find(key);
        if (found == cache.end())
        {
            int device = 0, sms = 0, per_sm = 0;
            cuda_check(cudaGetDevice(&device));
            cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, 0));
            found = cache.emplace(key, 8u * static_cast<unsigned int>(sms) * static_cast<unsigned int>(per_sm > 0 ? per_sm : 1)).first;
        }
        return min(found->second, m_max_blocks);
    }

    // Fused-perception launch helpers; pointers are in binned particle order.
    void launch_fused_density(
        const ParticlePosition<DIM> *positions, const float *mass, float *volume, float *density_gradient,
        const uint32_t *bin_offsets, const BlockInfo *block_info, bool perceive_density_grad, cudaStream_t stream)
    {
        if (perceive_density_grad)
        {
            auto kernel = fused_perception::density_forward<DIM, BC, true>;
            constexpr int THREADS = fused_perception::DensityLayout<DIM, true>::THREADS;
            kernel<<<persistent_grid(kernel, THREADS), THREADS, 0, stream>>>(
                positions, mass, volume, density_gradient, bin_offsets, block_info, m_config);
        }
        else
        {
            auto kernel = fused_perception::density_forward<DIM, BC, false>;
            constexpr int THREADS = fused_perception::DensityLayout<DIM, false>::THREADS;
            kernel<<<persistent_grid(kernel, THREADS), THREADS, 0, stream>>>(
                positions, mass, volume, nullptr, bin_offsets, block_info, m_config);
        }
        cuda_check(cudaGetLastError());
    }

    template <typename Launch>
    static void dispatch_features(uint32_t features, Launch &&launch)
    {
        if (features == 4)
            launch(std::integral_constant<int, 4>{});
        else if (features == 8)
            launch(std::integral_constant<int, 8>{});
        else if (features == 16)
            launch(std::integral_constant<int, 16>{});
        else if (features == 32)
            launch(std::integral_constant<int, 32>{});
        else
            throw std::runtime_error("Fused perception supports exactly 4, 8, 16, or 32 state channels.");
    }

    void launch_fused_features(
        const ParticlePosition<DIM> *positions, const float *state, const float *volume,
        const float *density_gradient, const float *total_mass, const float *eps0,
        float *perception, float *gradient_values,
        float *gradient_sum, const uint32_t *bin_offsets, const BlockInfo *block_info, uint32_t features,
        bool perceive_density_grad, bool normalize_state_grad, bool normalize_density_grad,
        bool save_backward, cudaStream_t stream)
    {
        dispatch_features(features, [&](auto feature_count) {
            constexpr int FEATURES = decltype(feature_count)::value;
            constexpr int THREADS = fused_perception::ForwardLayout<DIM, FEATURES>::THREADS;
            auto kernel = fused_perception::forward<DIM, FEATURES, BC>;
            kernel<<<persistent_grid(kernel, THREADS), THREADS, 0, stream>>>(
                positions, state, volume, density_gradient, total_mass, eps0, perception, gradient_values, gradient_sum,
                bin_offsets, block_info, m_config,
                perceive_density_grad, normalize_state_grad, normalize_density_grad, save_backward);
        });
        cuda_check(cudaGetLastError());
    }

    void launch_fused_backward(
        const ParticlePosition<DIM> *positions, const float *volume, const float *grad_perception,
        const float *gradient_values, const float *gradient_sum, const float *eps0, float *incoming,
        float *grad_state, const uint32_t *bin_offsets, const BlockInfo *block_info, uint32_t features,
        bool perceive_density_grad, bool normalize_state_grad, cudaStream_t stream)
    {
        uint32_t output_dim = features * (2 + DIM) + (perceive_density_grad ? DIM : 0);
        dispatch_features(features, [&](auto feature_count) {
            constexpr int FEATURES = decltype(feature_count)::value;
            unsigned int particles = m_config.batch_size * m_config.particle_count;
            constexpr int PREPARE_THREADS = fused_perception::BACKWARD_PREPARE_THREADS;
            unsigned int blocks = (particles * FEATURES + PREPARE_THREADS - 1) / PREPARE_THREADS;
            fused_perception::backward_prepare<DIM, FEATURES><<<blocks, PREPARE_THREADS, 0, stream>>>(
                grad_perception, gradient_values, gradient_sum, eps0,
                incoming, grad_state, particles, output_dim, m_config.eps, normalize_state_grad);
            cuda_check(cudaGetLastError());
            constexpr int THREADS = fused_perception::BackwardLayout<DIM, FEATURES>::THREADS;
            auto kernel = fused_perception::backward<DIM, FEATURES, BC>;
            kernel<<<persistent_grid(kernel, THREADS), THREADS, 0, stream>>>(
                positions, volume, incoming, grad_state, bin_offsets, block_info, m_config);
        });
        cuda_check(cudaGetLastError());
    }

private:
    // Device pointers of one standalone-operator launch; unused fields stay null.
    struct OperatorArgs
    {
        const ParticlePosition<DIM> *positions = nullptr;
        const uint32_t *bin_offsets = nullptr;
        const BlockInfo *block_info = nullptr;
        const float *rho = nullptr;
        const float *mass = nullptr;
        const float *state = nullptr;
        uint32_t feature_dim = 0;
        // Forward outputs.
        float *rho_out = nullptr;
        float *count_out = nullptr;
        float *blur_out = nullptr;
        float *gradient_out = nullptr;
        float *density_gradient_out = nullptr;
        float *moment_matrix_out = nullptr;
        // Backward: the operator output's cotangent, then the input adjoints.
        const float *dL_rho = nullptr;
        const float *dL_blur = nullptr;
        const float *dL_gradient = nullptr;
        const float *dL_density_gradient = nullptr;
        const float *dL_moment_matrix = nullptr;
        ParticlePosition<DIM> *grad_positions = nullptr;
        float *grad_rho = nullptr;
        float *grad_state = nullptr;
    };

    // Shape (batch_size, N, trailing...) of a per-particle array.
    std::vector<size_t> per_particle(std::initializer_list<size_t> trailing = {}) const
    {
        std::vector<size_t> shape{m_config.batch_size, m_config.particle_count};
        shape.insert(shape.end(), trailing);
        return shape;
    }

    // Checks the binned positions and snapshot arrays that every operator takes.
    OperatorArgs operator_args(const FloatArray3D &binned_positions, const IntArray1D &bin_offsets,
                               const BlockInfoArray &block_info) const
    {
        validate_array_shape(binned_positions, per_particle({m_dimension}), "binned_positions");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");
        OperatorArgs args;
        args.positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        args.bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        args.block_info = reinterpret_cast<const BlockInfo *>(block_info.data());
        return args;
    }

    // One CUDA block of THREADS per grid block, STRIDE particles per shared-memory tile.
    template <ComputationType TYPE, int STRIDE, int THREADS>
    void launch_forward(const OperatorArgs &a, size_t shared_floats) const
    {
        forward_kernel<DIM, TYPE, STRIDE, BC><<<m_max_blocks, THREADS, shared_floats * sizeof(float)>>>(
            a.positions, a.rho, a.mass, a.state, a.rho_out, a.count_out, a.blur_out, a.gradient_out,
            a.density_gradient_out, a.moment_matrix_out, a.feature_dim, a.bin_offsets, a.block_info, m_config);
        cuda_check(cudaGetLastError());
    }

    // DS_ONLY computes only the state adjoint (blur and gradient).
    template <ComputationType TYPE, int STRIDE, int THREADS, bool DS_ONLY = false>
    void launch_backward(const OperatorArgs &a, size_t shared_floats) const
    {
        backward_kernel<DIM, TYPE, STRIDE, BC, DS_ONLY><<<m_max_blocks, THREADS, shared_floats * sizeof(float)>>>(
            a.positions, a.rho, a.mass, a.state, a.dL_rho, nullptr, a.dL_blur, a.dL_gradient,
            a.dL_density_gradient, a.dL_moment_matrix, a.grad_positions, a.grad_rho, a.grad_state,
            a.feature_dim, a.bin_offsets, a.block_info, m_config);
        cuda_check(cudaGetLastError());
    }

public:
    // Standalone SPH operators on binned particles. Shapes: positions (B, N, DIM),
    // mass, rho and count (B, N), states (B, N, C), bin_offsets (B * cells + 1),
    // block_info (max_blocks, 4). Shared-memory sizes are floats per tile slot.

    void density_forward(FloatArray3D binned_positions, FloatArray2D mass, FloatArray2D rho,
                         IntArray1D bin_offsets, BlockInfoArray block_info)
    {
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(mass, per_particle(), "mass");
        validate_array_shape(rho, per_particle(), "rho");
        args.mass = mass.data();
        args.rho_out = rho.data();
        launch_forward<ComputationType::Density, 128, 128>(args, 128 * 2); // output, neighbor mass
    }

    void count_forward(FloatArray3D binned_positions, FloatArray2D count,
                       IntArray1D bin_offsets, BlockInfoArray block_info)
    {
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(count, per_particle(), "count");
        args.count_out = count.data();
        launch_forward<ComputationType::Count, 128, 128>(args, 128); // output
    }

    void blur_forward(FloatArray3D binned_positions, FloatArray2D rho, FloatArray2D mass, FloatArray3D state_in,
                      FloatArray3D state_out, IntArray1D bin_offsets, BlockInfoArray block_info)
    {
        uint32_t feature_dim = state_in.shape(2);
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(rho, per_particle(), "rho");
        validate_array_shape(mass, per_particle(), "mass");
        validate_array_shape(state_in, per_particle({feature_dim}), "state_in");
        validate_array_shape(state_out, per_particle({feature_dim}), "state_out");
        args.rho = rho.data();
        args.mass = mass.data();
        args.state = state_in.data();
        args.blur_out = state_out.data();
        args.feature_dim = feature_dim;
        // output, neighbor states, neighbor volume and mass
        launch_forward<ComputationType::Blur, 64, 256>(args, 64 * (feature_dim + feature_dim + 2));
    }

    void gradient_forward(FloatArray3D binned_positions, FloatArray2D rho, FloatArray2D mass, FloatArray3D state_in,
                          FloatArray4D state_gradient_out, IntArray1D bin_offsets, BlockInfoArray block_info)
    {
        uint32_t feature_dim = state_in.shape(2);
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(rho, per_particle(), "rho");
        validate_array_shape(mass, per_particle(), "mass");
        validate_array_shape(state_in, per_particle({feature_dim}), "state_in");
        validate_array_shape(state_gradient_out, per_particle({feature_dim, DIM}), "state_gradient_out");
        args.rho = rho.data();
        args.mass = mass.data();
        args.state = state_in.data();
        args.gradient_out = state_gradient_out.data();
        args.feature_dim = feature_dim;
        // output, neighbor and local states, neighbor volume and mass
        launch_forward<ComputationType::Gradient, 64, 256>(args, 64 * (feature_dim * DIM + feature_dim + feature_dim + 2));
    }

    void density_gradient_forward(FloatArray3D binned_positions, FloatArray2D mass, FloatArray3D density_gradient_out,
                                  IntArray1D bin_offsets, BlockInfoArray block_info)
    {
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(mass, per_particle(), "mass");
        validate_array_shape(density_gradient_out, per_particle({DIM}), "density_gradient_out");
        args.mass = mass.data();
        args.density_gradient_out = density_gradient_out.data();
        launch_forward<ComputationType::DensityGradient, 128, 128>(args, 128 * (DIM + 1)); // output, neighbor mass
    }

    void moment_matrix_forward(FloatArray3D binned_positions, FloatArray2D rho, FloatArray2D mass,
                               FloatArray4D moment_matrix_out, IntArray1D bin_offsets, BlockInfoArray block_info)
    {
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(rho, per_particle(), "rho");
        validate_array_shape(mass, per_particle(), "mass");
        validate_array_shape(moment_matrix_out, per_particle({DIM, DIM}), "moment_matrix_out");
        args.rho = rho.data();
        args.mass = mass.data();
        args.moment_matrix_out = moment_matrix_out.data();
        launch_forward<ComputationType::MomentMatrix, 128, 128>(args, 128 * (DIM * DIM + 2)); // output, neighbor volume and mass
    }

    // Backward operators: the dL* flags select the input adjoints. Every adjoint
    // pointer is passed; only the requested ones are shape-checked.

    void density_backward(FloatArray3D binned_positions, FloatArray2D mass, FloatArray2D dL_rho_out, FloatArray3D grad_x,
                          IntArray1D bin_offsets, BlockInfoArray block_info, bool dLdx = true)
    {
        if (!dLdx)
            return;
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(mass, per_particle(), "mass");
        validate_array_shape(dL_rho_out, per_particle(), "grad_output_rho");
        validate_array_shape(grad_x, per_particle({m_dimension}), "grad_x");
        args.mass = mass.data();
        args.dL_rho = dL_rho_out.data();
        args.grad_positions = reinterpret_cast<ParticlePosition<DIM> *>(grad_x.data());
        launch_backward<ComputationType::Density, 128, 128>(args, 128 * (1 + 1 + 2 + DIM));
    }

    void blur_backward(FloatArray3D binned_positions, FloatArray2D rho, FloatArray2D mass, FloatArray3D state_in,
                       FloatArray3D dL_blur_state_out, FloatArray3D grad_x, FloatArray3D grad_s, FloatArray2D grad_rho,
                       IntArray1D bin_offsets, BlockInfoArray block_info,
                       bool dLdx = true, bool dLds = true, bool dLdrho = true)
    {
        if (!dLdx && !dLds && !dLdrho)
            return;
        uint32_t feature_dim = state_in.shape(2);
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(mass, per_particle(), "mass");
        validate_array_shape(rho, per_particle(), "rho");
        validate_array_shape(state_in, per_particle({feature_dim}), "state_in");
        validate_array_shape(dL_blur_state_out, per_particle({feature_dim}), "dL_blur_state_out");
        if (dLdx)
            validate_array_shape(grad_x, per_particle({m_dimension}), "grad_x");
        if (dLds)
            validate_array_shape(grad_s, per_particle({feature_dim}), "grad_s");
        if (dLdrho)
            validate_array_shape(grad_rho, per_particle(), "grad_rho");
        args.rho = rho.data();
        args.mass = mass.data();
        args.state = state_in.data();
        args.dL_blur = dL_blur_state_out.data();
        args.grad_positions = reinterpret_cast<ParticlePosition<DIM> *>(grad_x.data());
        args.grad_rho = grad_rho.data();
        args.grad_state = grad_s.data();
        args.feature_dim = feature_dim;
        if (dLds && !dLdx && !dLdrho)
            launch_backward<ComputationType::Blur, 64, 256, true>(args, 64 * (feature_dim * 2 + 2));
        else
            launch_backward<ComputationType::Blur, 64, 256>(args, 64 * (feature_dim * 5 + DIM + 5));
    }

    void density_gradient_backward(FloatArray3D binned_positions, FloatArray2D mass, FloatArray3D dL_density_gradient_out,
                                   FloatArray3D grad_x, IntArray1D bin_offsets, BlockInfoArray block_info, bool dLdx = true)
    {
        if (!dLdx)
            return;
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(mass, per_particle(), "mass");
        validate_array_shape(dL_density_gradient_out, per_particle({DIM}), "dL_density_gradient_out");
        validate_array_shape(grad_x, per_particle({m_dimension}), "grad_x");
        args.mass = mass.data();
        args.dL_density_gradient = dL_density_gradient_out.data();
        args.grad_positions = reinterpret_cast<ParticlePosition<DIM> *>(grad_x.data());
        launch_backward<ComputationType::DensityGradient, 128, 128>(args, 128 * (DIM + DIM + DIM + 2));
    }

    void gradient_backward(FloatArray3D binned_positions, FloatArray2D rho, FloatArray2D mass, FloatArray3D state_in,
                           FloatArray4D dL_grad_state_out, FloatArray3D grad_x, FloatArray3D grad_s, FloatArray2D grad_rho,
                           IntArray1D bin_offsets, BlockInfoArray block_info,
                           bool dLdx = true, bool dLds = true, bool dLdrho = true)
    {
        if (!dLdx && !dLds && !dLdrho)
            return;
        uint32_t feature_dim = state_in.shape(2);
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(mass, per_particle(), "mass");
        validate_array_shape(rho, per_particle(), "rho");
        validate_array_shape(state_in, per_particle({feature_dim}), "state_in");
        validate_array_shape(dL_grad_state_out, per_particle({feature_dim, DIM}), "dL_grad_state_out");
        if (dLdx)
            validate_array_shape(grad_x, per_particle({m_dimension}), "grad_x");
        if (dLds)
            validate_array_shape(grad_s, per_particle({feature_dim}), "grad_s");
        if (dLdrho)
            validate_array_shape(grad_rho, per_particle(), "grad_rho");
        args.rho = rho.data();
        args.mass = mass.data();
        args.state = state_in.data();
        args.dL_gradient = dL_grad_state_out.data();
        args.grad_positions = reinterpret_cast<ParticlePosition<DIM> *>(grad_x.data());
        args.grad_rho = grad_rho.data();
        args.grad_state = grad_s.data();
        args.feature_dim = feature_dim;
        if (dLds && !dLdx && !dLdrho)
            launch_backward<ComputationType::Gradient, 32, 256, true>(args, 32 * (feature_dim * (2 * DIM + 1) + 4));
        else
            launch_backward<ComputationType::Gradient, 32, 256>(args, 32 * (feature_dim * (3 + 2 * DIM) + DIM + 5));
    }

    void moment_matrix_backward(FloatArray3D binned_positions, FloatArray2D rho, FloatArray2D mass,
                                FloatArray4D dL_moment_matrix_out, FloatArray3D grad_x, FloatArray2D grad_rho,
                                IntArray1D bin_offsets, BlockInfoArray block_info, bool dLdx = true, bool dLdrho = true)
    {
        if (!dLdx && !dLdrho)
            return;
        OperatorArgs args = operator_args(binned_positions, bin_offsets, block_info);
        validate_array_shape(mass, per_particle(), "mass");
        validate_array_shape(rho, per_particle(), "rho");
        validate_array_shape(dL_moment_matrix_out, per_particle({DIM, DIM}), "dL_moment_matrix_out");
        if (dLdx)
            validate_array_shape(grad_x, per_particle({m_dimension}), "grad_x");
        if (dLdrho)
            validate_array_shape(grad_rho, per_particle(), "grad_rho");
        args.rho = rho.data();
        args.mass = mass.data();
        args.dL_moment_matrix = dL_moment_matrix_out.data();
        args.grad_positions = reinterpret_cast<ParticlePosition<DIM> *>(grad_x.data());
        args.grad_rho = grad_rho.data();
        launch_backward<ComputationType::MomentMatrix, 128, 128>(args, 128 * (DIM * DIM * 2 + 2 + DIM + 3));
    }
};

template <BoundaryCondition BC>
class HashGrid2D : public HashGrid<2, BC>
{
public:
    HashGrid2D(int grid_size_x, int grid_size_y, float eps, uint32_t particle_count, uint32_t batch_size, uint32_t max_particles_per_block)
    {
        this->_init({grid_size_x, grid_size_y}, eps, particle_count, batch_size, max_particles_per_block);
    }
};

using HashGrid2DPeriodic = HashGrid2D<BoundaryCondition::PERIODIC>;
using HashGrid2DBox = HashGrid2D<BoundaryCondition::CLAMP>;

template <BoundaryCondition BC>
class HashGrid3D : public HashGrid<3, BC>
{
public:
    HashGrid3D(int grid_size_x, int grid_size_y, int grid_size_z, float eps, uint32_t particle_count, uint32_t batch_size, uint32_t max_particles_per_block)
    {
        this->_init({grid_size_x, grid_size_y, grid_size_z}, eps, particle_count, batch_size, max_particles_per_block);
    }

    int grid_size_z() const { return this->m_config.grid_size[2]; }
};

using HashGrid3DPeriodic = HashGrid3D<BoundaryCondition::PERIODIC>;
using HashGrid3DBox = HashGrid3D<BoundaryCondition::CLAMP>;

// One thread per symmetric DIM x DIM matrix; see safe_inverse_symmetric.
template <int DIM>
__global__ void safe_inverse_kernel(const float *__restrict__ in_matrix, float *__restrict__ out_matrix,
                                    uint32_t count, float tolerance)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    float matrix[DIM * DIM];
    if constexpr (DIM == 2)
    {
        float4 m = reinterpret_cast<const float4 *>(in_matrix)[idx];
        matrix[0] = m.x;
        matrix[1] = m.y;
        matrix[2] = m.z;
        matrix[3] = m.w;
    }
    else
    {
        for (int k = 0; k < DIM * DIM; ++k)
            matrix[k] = in_matrix[idx * DIM * DIM + k];
    }
    safe_inverse_symmetric<DIM>(matrix, tolerance);
    if constexpr (DIM == 2)
        reinterpret_cast<float4 *>(out_matrix)[idx] = make_float4(matrix[0], matrix[1], matrix[2], matrix[3]);
    else
    {
        for (int k = 0; k < DIM * DIM; ++k)
            out_matrix[idx * DIM * DIM + k] = matrix[k];
    }
}

void safe_inv_sym_mat(
    FloatArray4D in_matrix,  // Shape (batch_size, N, dim, dim)
    FloatArray4D out_matrix, // Shape (batch_size, N, dim, dim)
    float tol = 1e-3f)
{
    uint32_t batch_size = in_matrix.shape(0);
    uint32_t N = in_matrix.shape(1);
    uint32_t dim = in_matrix.shape(2);
    if (dim != 2 && dim != 3)
        throw std::runtime_error("safe_inverse only supports 2x2 or 3x3 matrices.");
    validate_array_shape(in_matrix, {batch_size, N, dim, dim}, "in_matrix");
    validate_array_shape(out_matrix, {batch_size, N, dim, dim}, "out_matrix");

    uint32_t count = batch_size * N;
    unsigned int blocks = (count + 255) / 256;
    if (dim == 2)
        safe_inverse_kernel<2><<<blocks, 256>>>(in_matrix.data(), out_matrix.data(), count, tol);
    else
        safe_inverse_kernel<3><<<blocks, 256>>>(in_matrix.data(), out_matrix.data(), count, tol);
    cuda_check(cudaGetLastError());
}

// Macro to add common properties and methods
#define ADD_COMMON_GRID_BINDINGS(grid_class, GridType, pos_shape)                                                                 \
    grid_class                                                                                                                    \
        .def("fused_perception_forward", &GridType::fused_perception_forward,                                                    \
             "positions"_a, "state"_a, "mass"_a.none(), "eps0"_a, "snapshot"_a, "binned_positions"_a,                             \
             "binned_state"_a, "perception"_a, "workspace"_a, "perceive_density_grad"_a, "normalize_state_grad"_a,               \
             "normalize_density_grad"_a, "save_backward"_a, "stream"_a = 0)                                                      \
        .def("fused_perception_backward", &GridType::fused_perception_backward,                                                  \
             "binned_positions"_a, "workspace"_a, "grad_perception"_a.none(), "grad_binned_positions"_a.none(),                  \
             "grad_binned_state"_a.none(), "eps0"_a, "snapshot"_a, "scratch"_a, "grad_positions"_a.none(),                       \
             "grad_state"_a.none(), "features"_a, "perceive_density_grad"_a, "normalize_state_grad"_a, "has_mass"_a,             \
             "stream"_a = 0)                                                                                                      \
        .def("fused_workspace_size", &GridType::fused_workspace_size,                                                            \
             "features"_a, "density_grad"_a, "normalize_state_grad"_a, "save_backward"_a, "has_mass"_a)                          \
        .def_prop_ro("fused_snapshot_words", &GridType::fused_snapshot_words, "Fused snapshot buffer length (int32 words)")      \
        .def_prop_ro("grid_size_x", &GridType::grid_size_x, "Grid size in x dimension")                                           \
        .def_prop_ro("grid_size_y", &GridType::grid_size_y, "Grid size in y dimension")                                           \
        .def_prop_ro("eps", &GridType::eps, "Grid cell spacing")                                                                  \
        .def_prop_ro("particle_count", &GridType::particle_count, "Number of particles per batch")                                \
        .def_prop_ro("batch_size", &GridType::batch_size, "Batch size")                                                           \
        .def_prop_ro("max_blocks", &GridType::max_blocks, "Max blocks needed for cuda kernels")                                   \
        .def_prop_ro("max_particles_per_block", &GridType::max_particles_per_block, "Max particles per block")                    \
        .def("initialize", &GridType::initialize, "positions"_a, "permutation"_a, "bin_offsets"_a, "block_info"_a,                \
             "Initialize the grid with particle positions. 'positions' should have shape " pos_shape                              \
             " 'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                    \
             "'permutation' should have shape (batch_size, N). "                                                                  \
             "'block_info' should have shape (max_blocks, 4).")                                                                   \
                                                                                                                                  \
        .def("bin_particles", &GridType::bin_particles,                                                                           \
               "input_positions"_a, "input_state"_a, "input_mass"_a, "permutation"_a, "output_positions"_a, "output_state"_a,      \
               "output_mass"_a, "Bin particle positions, states, and masses according to the computed permutation. "              \
             "'input_positions' and 'output_positions' should have shape " pos_shape ". "                                         \
               "'input_state' and 'output_state' should have shape (batch_size, N, feature_dim). "                                  \
               "'input_mass' and 'output_mass' should have shape (batch_size, N).")                                                 \
        .def("backward_bin_particles", &GridType::backward_bin_particles,                                                         \
               "grad_output_positions"_a, "grad_output_state"_a, "grad_output_mass"_a, "permutation"_a, "grad_input_positions"_a, \
               "grad_input_state"_a, "grad_input_mass"_a,                                                                           \
               "Compute gradients for binned particle positions, states, and masses according to the permutation. "                \
             "'grad_output_positions' and 'grad_input_positions' should have shape " pos_shape ". "                               \
               "'grad_output_state' and 'grad_input_state' should have shape (batch_size, N, feature_dim). "                        \
               "'grad_output_mass' and 'grad_input_mass' should have shape (batch_size, N).")                                       \
                                                                                                                                  \
        .def("density_forward", &GridType::density_forward,                                                                       \
             "binned_positions"_a, "mass"_a, "rho"_a, "bin_offsets"_a, "block_info"_a,                                            \
             "Compute local particle density based on binned positions. "                                                         \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'mass' should have shape (batch_size, N). "                                                                         \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'Rho' should have shape (batch_size, N). "                                                                          \
             "'block_info' should have shape (max_blocks, 4).")                                                                   \
                                                                                                                                  \
        .def("count_forward", &GridType::count_forward,                                                                           \
             "binned_positions"_a, "count"_a, "bin_offsets"_a, "block_info"_a,                                                    \
             "Compute local particle count based on binned positions. "                                                           \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'count' should have shape (batch_size, N). "                                                                        \
             "'block_info' should have shape (max_blocks, 4).")                                                                   \
        .def("blur_forward", &GridType::blur_forward,                                                                             \
             "binned_positions"_a, "rho"_a, "mass"_a, "state_in"_a, "state_out"_a, "bin_offsets"_a, "block_info"_a,               \
             "Compute local particle state blur based on binned positions and input density. "                                    \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'rho' should have shape (batch_size, N). "                                                                          \
             "'mass' should have shape (batch_size, N). "                                                                         \
             "'state_in' and 'state_out' should have shape (batch_size, N, feature_dim)."                                         \
             "'block_info' should have shape (max_blocks, 4).")                                                                   \
        .def("gradient_forward", &GridType::gradient_forward,                                                                     \
             "binned_positions"_a, "rho"_a, "mass"_a, "state_in"_a, "state_gradient_out"_a, "bin_offsets"_a, "block_info"_a,      \
             "Compute local particle state gradient based on binned positions and input density. "                                \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'rho' should have shape (batch_size, N). "                                                                          \
             "'mass' should have shape (batch_size, N). "                                                                         \
             "'state_in' should have shape (batch_size, N, feature_dim). "                                                        \
             "'state_gradient_out' should have shape (batch_size, N, feature_dim, DIM)."                                          \
             "'block_info' should have shape (max_blocks, 4).")                                                                   \
        .def("density_gradient_forward", &GridType::density_gradient_forward,                                                     \
             "binned_positions"_a, "mass"_a, "density_gradient_out"_a, "bin_offsets"_a, "block_info"_a,                           \
             "Compute local particle density gradient based on binned positions and input density. "                              \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'mass' should have shape (batch_size, N). "                                                                         \
             "'density_gradient_out' should have shape (batch_size, N, DIM)."                                                     \
             "'block_info' should have shape (max_blocks, 4).")                                                                   \
        .def("moment_matrix_forward", &GridType::moment_matrix_forward,                                                           \
             "binned_positions"_a, "rho"_a, "mass"_a, "moment_matrix_out"_a, "bin_offsets"_a, "block_info"_a,                     \
             "Compute local particle moment matrix based on binned positions and input density. "                                 \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'rho' should have shape (batch_size, N). "                                                                          \
             "'mass' should have shape (batch_size, N). "                                                                         \
             "'moment_matrix_out' should have shape (batch_size, N, DIM, DIM)."                                                   \
             "'block_info' should have shape (max_blocks, 4).")                                                                   \
                                                                                                                                  \
        .def("density_backward", &GridType::density_backward,                                                                     \
             "binned_positions"_a, "mass"_a, "dL_rho_out"_a, "grad_x"_a, "bin_offsets"_a, "block_info"_a, nb::arg("dLdx") = true, \
             "Compute backward gradient for input positions based on gradient of output density. "                                \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'dL_rho_out' should have shape (batch_size, N). "                                                                   \
             "'grad_x' should have shape (batch_size, N, DIM). "                                                                  \
             "'block_info' should have shape (max_blocks, 4). "                                                                   \
             "'dLdx' indicates whether to compute gradient w.r.t. position.")                                                     \
        .def("blur_backward", &GridType::blur_backward,                                                                           \
             "binned_positions"_a, "rho"_a, "mass"_a, "state_in"_a, "dL_blur_state_out"_a, "grad_x"_a,                            \
             "grad_s"_a, "grad_rho"_a, "bin_offsets"_a, "block_info"_a, nb::arg("dLdx") = true,                                   \
             nb::arg("dLds") = true, nb::arg("dLdrho") = true,                                                                    \
             "Compute backward gradient for input positions, state, and density based on gradient of output state. "              \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'rho' should have shape (batch_size, N). "                                                                          \
             "'mass' should have shape (batch_size, N). "                                                                         \
             "'state_in' should have shape (batch_size, N, feature_dim). "                                                        \
             "'dL_blur_state_out' should have shape (batch_size, N, feature_dim). "                                               \
             "'grad_x' should have shape (batch_size, N, DIM). "                                                                  \
             "'grad_s' should have shape (batch_size, N, feature_dim). "                                                          \
             "'grad_rho' should have shape (batch_size, N). "                                                                     \
             "'block_info' should have shape (max_blocks, 4). "                                                                   \
             "'dLdx', 'dLds', and 'dLdrho' indicate whether to compute"                                                           \
             " gradient w.r.t. position, state, and density respectively.")                                                       \
        .def("density_gradient_backward", &GridType::density_gradient_backward,                                                   \
             "binned_positions"_a, "mass"_a, "dL_density_gradient_out"_a, "grad_x"_a, "bin_offsets"_a,                            \
             "block_info"_a, nb::arg("dLdx") = true,                                                                              \
             "Compute backward gradient for input positions based on gradient of output density."                                 \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'dL_density_gradient_out' should have shape (batch_size, N, DIM). "                                                 \
             "'grad_x' should have shape (batch_size, N, DIM). "                                                                  \
             "'block_info' should have shape (max_blocks, 4). "                                                                   \
             "'dLdx' indicates whether to compute gradient w.r.t. position.")                                                     \
        .def("gradient_backward", &GridType::gradient_backward,                                                                   \
             "binned_positions"_a, "rho"_a, "mass"_a, "state_in"_a, "dL_grad_state_out"_a, "grad_x"_a,                            \
             "grad_s"_a, "grad_rho"_a, "bin_offsets"_a, "block_info"_a, nb::arg("dLdx") = true,                                   \
             nb::arg("dLds") = true, nb::arg("dLdrho") = true,                                                                    \
             "Compute backward gradient for input positions, state, and density based on gradient of output state. "              \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'rho' should have shape (batch_size, N). "                                                                          \
             "'mass' should have shape (batch_size, N). "                                                                         \
             "'state_in' should have shape (batch_size, N, feature_dim). "                                                        \
             "'dL_grad_state_out' should have shape (batch_size, N, feature_dim). "                                               \
             "'grad_x' should have shape (batch_size, N, DIM). "                                                                  \
             "'grad_s' should have shape (batch_size, N, feature_dim). "                                                          \
             "'grad_rho' should have shape (batch_size, N). "                                                                     \
             "'block_info' should have shape (max_blocks, 4). "                                                                   \
             "'dLdx', 'dLds', and 'dLdrho' indicate whether to compute"                                                           \
             " gradient w.r.t. position, state, and density respectively.")                                                       \
        .def("moment_matrix_backward", &GridType::moment_matrix_backward,                                                         \
             "binned_positions"_a, "rho"_a, "mass"_a, "dL_moment_matrix_out"_a, "grad_x"_a,                                       \
             "grad_rho"_a, "bin_offsets"_a, "block_info"_a, nb::arg("dLdx") = true,                                               \
             nb::arg("dLdrho") = true,                                                                                            \
             "Compute backward gradient for input positions and density based on gradient of output moment matrix. "              \
             "'binned_positions' should have shape " pos_shape ". "                                                               \
             "'bin_offsets' should have shape (batch_size, cell_count + 1). "                                                     \
             "'rho' should have shape (batch_size, N). "                                                                          \
             "'mass' should have shape (batch_size, N). "                                                                         \
             "'dL_moment_matrix_out' should have shape (batch_size, N, DIM, DIM). "                                               \
             "'grad_x' should have shape (batch_size, N, DIM). "                                                                  \
             "'grad_rho' should have shape (batch_size, N). "                                                                     \
             "'block_info' should have shape (max_blocks, 4). "                                                                   \
             "'dLdx' and 'dLdrho' indicate whether to compute"                                                                    \
             " gradient w.r.t. position and density respectively.")

NB_MODULE(sph_cuda, m)
{
    // 2D HashGrid
    auto grid2d_periodic = nb::class_<HashGrid2DPeriodic>(m, "HashGrid2DPeriodic")
                               .def(nb::init<int, int, float, uint32_t, uint32_t, uint32_t>(),
                                    "grid_size_x"_a, "grid_size_y"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                                    "Initialize a 2D hash grid with given resolution, cell size, particle count, and batch size.");
    ADD_COMMON_GRID_BINDINGS(grid2d_periodic, HashGrid2DPeriodic, "(batch_size, N, 2)");

    auto grid2d_box = nb::class_<HashGrid2DBox>(m, "HashGrid2DBox")
                          .def(nb::init<int, int, float, uint32_t, uint32_t, uint32_t>(),
                               "grid_size_x"_a, "grid_size_y"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                               "Initialize a 2D hash grid with given resolution, cell size, particle count, and batch size.");
    ADD_COMMON_GRID_BINDINGS(grid2d_box, HashGrid2DBox, "(batch_size, N, 2)");

    // 3D HashGrid
    auto grid3d_periodic = nb::class_<HashGrid3DPeriodic>(m, "HashGrid3DPeriodic")
                               .def(nb::init<int, int, int, float, uint32_t, uint32_t, uint32_t>(),
                                    "grid_size_x"_a, "grid_size_y"_a, "grid_size_z"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                                    "Initialize a 3D hash grid with given resolution, cell size, particle count, and batch size.")
                               .def_prop_ro("grid_size_z", &HashGrid3DPeriodic::grid_size_z, "Grid size in z dimension");
    ADD_COMMON_GRID_BINDINGS(grid3d_periodic, HashGrid3DPeriodic, "(batch_size, N, 3)");

    auto grid3d_box = nb::class_<HashGrid3DBox>(m, "HashGrid3DBox")
                          .def(nb::init<int, int, int, float, uint32_t, uint32_t, uint32_t>(),
                               "grid_size_x"_a, "grid_size_y"_a, "grid_size_z"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                               "Initialize a 3D hash grid with given resolution, cell size, particle count, and batch size.")
                          .def_prop_ro("grid_size_z", &HashGrid3DBox::grid_size_z, "Grid size in z dimension");
    ADD_COMMON_GRID_BINDINGS(grid3d_box, HashGrid3DBox, "(batch_size, N, 3)");

    m.def("safe_inv_sym_mat", &safe_inv_sym_mat, "in_matrix"_a, "out_matrix"_a, "tol"_a,
          "Compute the safe inverse of a batch of 2x2 or 3x3 symmetric matrices."
          "If the matrix is near-singular, the output will be the identity matrix.");
}
