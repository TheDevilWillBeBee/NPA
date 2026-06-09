#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdexcept>
#include "utils.cuh"

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

template <int DIM, BoundaryCondition BC, KernelStrategy ST>
__global__ void grid_count_particles(
    const ParticlePosition<DIM> * __restrict__ positions,
    uint32_t * __restrict__ cell_counts,
    uint32_t * __restrict__ blocks_per_cell,
    GridConfig<DIM, BC, ST> config)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    uint32_t idx = batch_idx * config.particle_count + tid;
    if (tid >= config.particle_count)
        return;
    if (batch_idx >= config.batch_size)
        return;

    ParticlePosition<DIM> pos = positions[idx];

    int cell[DIM];
    config.pos2cell(pos, cell);
    uint32_t hash = config.cell2hash(cell);
    uint32_t cell_idx = batch_idx * config.cell_count + hash;

    uint32_t prev_count = atomicAdd(&cell_counts[cell_idx], 1U);
    if constexpr (ST == KernelStrategy::GridBased)
        if (prev_count % config.MAX_PARTICLES_PER_BLOCK == 0)
            atomicAdd(&blocks_per_cell[cell_idx], 1U);
}

template <int DIM, BoundaryCondition BC, KernelStrategy ST>
__global__ void grid_compute_permutation(
    const ParticlePosition<DIM> * __restrict__ positions,
    const uint32_t * __restrict__ bin_offsets,
    uint32_t * __restrict__ bin_write_indices,
    uint32_t * __restrict__ permutation,
    GridConfig<DIM, BC, ST> config)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    uint32_t idx = batch_idx * config.particle_count + tid;
    if (tid >= config.particle_count)
        return;
    if (batch_idx >= config.batch_size)
        return;

    ParticlePosition<DIM> pos = positions[idx];

    int cell[DIM];
    config.pos2cell(pos, cell);
    uint32_t hash = config.cell2hash(cell);
    uint32_t write_idx = atomicAdd(&bin_write_indices[batch_idx * config.cell_count + hash], 1U);

    permutation[idx] = write_idx + bin_offsets[batch_idx * config.cell_count + hash] - batch_idx * config.particle_count;
}

template <int DIM, BoundaryCondition BC, KernelStrategy ST>
__global__ void apply_permutation(
    const ParticlePosition<DIM> * __restrict__ input_positions,
    const float * __restrict__ input_state,
    const float * __restrict__ input_mass,
    const uint32_t * __restrict__ permutation,
    ParticlePosition<DIM> * __restrict__ output_positions,
    float * __restrict__ output_state,
    float * __restrict__ output_mass,
    uint32_t feature_dim,
    GridConfig<DIM, BC, ST> config)
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
    output_mass[permuted_idx] = input_mass[idx];
    uint32_t f = 0;
    // Vectorized float4 fast path when feature_dim is a multiple of 4
    if ((feature_dim & 3) == 0)
    {
        const float4 *src4 = reinterpret_cast<const float4 *>(&input_state[idx * feature_dim]);
        float4 *dst4 = reinterpret_cast<float4 *>(&output_state[permuted_idx * feature_dim]);
        for (; f < feature_dim; f += 4)
            dst4[f >> 2] = src4[f >> 2];
    }
    else
    {
        for (; f < feature_dim; ++f)
            output_state[permuted_idx * feature_dim + f] = input_state[idx * feature_dim + f];
    }
}

template <int DIM, BoundaryCondition BC, KernelStrategy ST>
__global__ void backward_apply_permutation(
    const ParticlePosition<DIM> * __restrict__ grad_output_positions,
    const float * __restrict__ grad_output_state,
    const float * __restrict__ grad_output_mass,
    const uint32_t * __restrict__ permutation,
    ParticlePosition<DIM> * __restrict__ grad_input_positions,
    float * __restrict__ grad_input_state,
    float * __restrict__ grad_input_mass,
    uint32_t feature_dim,
    GridConfig<DIM, BC, ST> config)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    if (tid >= config.particle_count)
        return;
    if (batch_idx >= config.batch_size)
        return;

    uint32_t idx = batch_idx * config.particle_count + tid;

    uint32_t permuted_idx = permutation[idx] + batch_idx * config.particle_count;

    // Backward pass for positions: grad_input_positions[idx] = grad_output_positions[permuted_idx]
    grad_input_positions[idx] = grad_output_positions[permuted_idx];
    grad_input_mass[idx] = grad_output_mass[permuted_idx];

    // Backward pass for state: grad_input_state[idx * feature_dim + f] = grad_output_state[permuted_idx * feature_dim + f]
    uint32_t f = 0;
    // Vectorized float4 fast path when feature_dim is a multiple of 4
    if ((feature_dim & 3) == 0)
    {
        const float4 *src4 = reinterpret_cast<const float4 *>(&grad_output_state[permuted_idx * feature_dim]);
        float4 *dst4 = reinterpret_cast<float4 *>(&grad_input_state[idx * feature_dim]);
        for (; f < feature_dim; f += 4)
            dst4[f >> 2] = src4[f >> 2];
    }
    else
    {
        for (; f < feature_dim; ++f)
            grad_input_state[idx * feature_dim + f] = grad_output_state[permuted_idx * feature_dim + f];
    }
}
// Kernel to assign block information
template <int DIM, BoundaryCondition BC, KernelStrategy ST = KernelStrategy::GridBased>
__global__ void assign_block_info(
    const uint32_t * __restrict__ cell_counts,   // Shape [batch_size * cell_count]
    const uint32_t * __restrict__ block_offsets, // Cumulative sum of blocks per cell (batch_size * cell_count + 1)
    BlockInfo * __restrict__ block_info,         // Output: block information # Shape [batch_size * max_blocks_per_batch]
    GridConfig<DIM, BC, ST> config)
{
    uint32_t cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;

    if (cell_idx >= config.cell_count || batch_idx >= config.batch_size)
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

enum class ComputationType
{
    Count,           // 0
    Density,         // 1
    DensityGradient, // 2
    MomentMatrix,    // 3
    Blur,            // 4
    Gradient         // 5
};

template <int DIM, ComputationType COMP_TYPE, int MAX_FEATURES, BoundaryCondition BC = BoundaryCondition::PERIODIC, KernelStrategy ST = KernelStrategy::ParticleBased>
__global__ void vanilla_forward_kernel(
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
    const GridConfig<DIM, BC, ST> config)
{
    uint32_t batch_idx = blockIdx.y;
    uint32_t particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx >= config.particle_count || batch_idx >= config.batch_size)
        return;

    constexpr int MEM_SIZE_LOCAL_STATE = (COMP_TYPE == ComputationType::Gradient) ? MAX_FEATURES : 1;
    constexpr int MEM_SIZE_NEIGHBOR_STATE = (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient) ? MAX_FEATURES : 1;
    constexpr int MEM_SIZE_LOCAL_BUFFER = []()
    {
        if constexpr (COMP_TYPE == ComputationType::Blur)
            return MAX_FEATURES;
        else if constexpr (COMP_TYPE == ComputationType::Gradient)
            return MAX_FEATURES * DIM;
        else if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
            return DIM * DIM;
        else if constexpr (COMP_TYPE == ComputationType::DensityGradient)
            return DIM;
        else
            return 1;
    }();

    float *global_output_buffer;
    int output_dim;
    float coef;
    if constexpr (COMP_TYPE == ComputationType::Count)
    {
        global_output_buffer = &count_out[batch_idx * config.particle_count + particle_idx];
        output_dim = 1;
        coef = 1.0f;
    }
    else if constexpr (COMP_TYPE == ComputationType::Density)
    {
        global_output_buffer = &rho_out[batch_idx * config.particle_count + particle_idx];
        output_dim = 1;
        coef = config.smoothing_coef;
    }
    else if constexpr (COMP_TYPE == ComputationType::Blur)
    {
        global_output_buffer = &blur_state_out[(batch_idx * config.particle_count + particle_idx) * feature_dim];
        output_dim = feature_dim;
        coef = config.smoothing_coef;
    }
    else if constexpr (COMP_TYPE == ComputationType::Gradient)
    {
        global_output_buffer = &grad_state_out[(batch_idx * config.particle_count + particle_idx) * feature_dim * DIM];
        output_dim = feature_dim * DIM;
        coef = config.spiky_coef;
    }
    else if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
    {
        global_output_buffer = &moment_matrix_out[(batch_idx * config.particle_count + particle_idx) * DIM * DIM];
        output_dim = DIM * DIM;
        coef = config.spiky_coef;
    }
    else if constexpr (COMP_TYPE == ComputationType::DensityGradient)
    {
        global_output_buffer = &grad_density_out[(batch_idx * config.particle_count + particle_idx) * DIM];
        output_dim = DIM;
        coef = config.spiky_coef;
    }

    constexpr int chunk_types = []()
    {
        if constexpr (BC == BoundaryCondition::PERIODIC)
            return 2;
        else
            return 1;
    }();
    constexpr int NUM_NEIGHBOR_CELLS = (DIM == 2) ? 9 : 27;

    float s_i[MEM_SIZE_LOCAL_STATE];
    float s_j[MEM_SIZE_NEIGHBOR_STATE];
    float local_output_buffer[MEM_SIZE_LOCAL_BUFFER] = {0.0f};
    float rho_j, v_j; // Rho and specific volume for neighbor particle j
    float m_j;
    float default_mass = config.default_mass;
    float eps2 = config.eps * config.eps;
    ParticlePosition<DIM> p_i = read_position<DIM>(position_in, config.particle_count, batch_idx, particle_idx);

    if constexpr (COMP_TYPE == ComputationType::Gradient)
        for (uint32_t f = 0; f < feature_dim; ++f)
            s_i[f] = read_state(state_in, config.particle_count, feature_dim, batch_idx, particle_idx, f);

    int local_cell[DIM];
    int neighbor_cell[DIM];
    config.pos2cell(p_i, local_cell);
    uint32_t local_cell_idx = config.cell2hash(local_cell);

    // TODO: Switch to Morton Hashing for better memory access patterns
    // Change GridConfig accordingly
    for (int k = 0; k < NUM_NEIGHBOR_CELLS; k++)
    {
        if constexpr (DIM == 2)
        {
            neighbor_cell[0] = local_cell[0] + (k % 3 - 1);
            neighbor_cell[1] = local_cell[1] + (k / 3 - 1);
        }
        else if constexpr (DIM == 3)
        {
            neighbor_cell[0] = local_cell[0] + (k % 3 - 1);
            neighbor_cell[1] = local_cell[1] + ((k / 3) % 3 - 1);
            neighbor_cell[2] = local_cell[2] + (k / 9 - 1);
        }

        if (!config.is_in_bound(neighbor_cell))
            continue;

        config.apply_boundary(neighbor_cell);
        uint32_t neighbor_cell_idx = config.cell2hash(neighbor_cell);

        uint32_t remote_start = bin_offsets[batch_idx * config.cell_count + neighbor_cell_idx] - batch_idx * config.particle_count;
        uint32_t remote_end = bin_offsets[batch_idx * config.cell_count + neighbor_cell_idx + 1] - batch_idx * config.particle_count;
        uint32_t remote_count = remote_end - remote_start;
        if (remote_count == 0)
            continue;

        for (uint32_t j = 0; j < remote_count; ++j)
        {
            ParticlePosition<DIM> p_j = read_position<DIM>(position_in, config.particle_count, batch_idx, remote_start + j);
            if (config.l2_distance(p_i, p_j) >= eps2)
                continue;

            if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::MomentMatrix)
            {
                rho_j = read_state(rho_in, config.particle_count, 1, batch_idx, remote_start + j, 0);
                m_j = read_mass(mass_in, config.particle_count, batch_idx, remote_start + j, default_mass);
                v_j = m_j / rho_j;
            }

            if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient)
                for (uint32_t f = 0; f < feature_dim; ++f)
                    s_j[f] = read_state(state_in, config.particle_count, feature_dim, batch_idx, remote_start + j, f);

            if constexpr (COMP_TYPE == ComputationType::Count)
                local_output_buffer[0] += 1.0f;

            if constexpr (COMP_TYPE == ComputationType::Density)
            {
                m_j = read_mass(mass_in, config.particle_count, batch_idx, remote_start + j, default_mass);
                ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                local_output_buffer[0] += smoothing_kernel(r_ij, config.eps) * m_j;
            }

            if constexpr (COMP_TYPE == ComputationType::Blur)
            {
                ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                float w = smoothing_kernel(r_ij, config.eps) * v_j;
                for (uint32_t f = 0; f < feature_dim; ++f)
                    local_output_buffer[f] += w * s_j[f];
            }

            if constexpr (COMP_TYPE == ComputationType::Gradient)
            {
                float w[DIM] = {0.0f};
                ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                spiky_kernel_inplace(r_ij, config.eps, v_j, w);
                for (uint32_t f = 0; f < feature_dim; ++f)
                    for (int d = 0; d < DIM; ++d)
                        local_output_buffer[f * DIM + d] += w[d] * (s_j[f] - s_i[f]);
            }

            if constexpr (COMP_TYPE == ComputationType::DensityGradient)
            {
                m_j = read_mass(mass_in, config.particle_count, batch_idx, remote_start + j, default_mass);
                ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                spiky_kernel_inplace(r_ij, config.eps, m_j, &local_output_buffer[0]);
            }

            if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
            {
                ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                moment_matrix_inplace(r_ij, config.eps, v_j, &local_output_buffer[0]);
            }
        }
    }

    for (int i = 0; i < output_dim; ++i)
        global_output_buffer[i] = local_output_buffer[i] * coef;
}

template <int DIM, ComputationType COMP_TYPE, int MAX_FEATURES,
          BoundaryCondition BC = BoundaryCondition::PERIODIC,
          KernelStrategy ST = KernelStrategy::GridBased,
          bool COMPUTE_DS_ONLY = false>
__global__ void vanilla_backward_kernel(
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
    const GridConfig<DIM, BC, ST> config)
{
    uint32_t batch_idx = blockIdx.y;
    uint32_t particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx >= config.particle_count || batch_idx >= config.batch_size)
        return;

    if constexpr (COMP_TYPE == ComputationType::Count)
        return; // No gradients to compute for count

    constexpr int MEM_SIZE_LOCAL_GRAD = []()
    {
        if constexpr (COMP_TYPE == ComputationType::Gradient)
            return MAX_FEATURES * DIM;
        else if constexpr (COMP_TYPE == ComputationType::Blur)
            return COMPUTE_DS_ONLY ? 1 : MAX_FEATURES;
        else if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
            return DIM * DIM;
        else if constexpr (COMP_TYPE == ComputationType::DensityGradient)
            return DIM;
        else
            return 1; // Density
    }();

    constexpr int MEM_SIZE_NEIGHBOR_GRAD = []()
    {
        if constexpr (COMP_TYPE == ComputationType::Gradient)
            return MAX_FEATURES * DIM;
        else if constexpr (COMP_TYPE == ComputationType::Blur)
            return MAX_FEATURES;
        else if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
            return DIM * DIM;
        else if constexpr (COMP_TYPE == ComputationType::DensityGradient)
            return DIM;
        else
            return 1;
    }();

    constexpr int MEM_SIZE_LOCAL_STATE = []()
    {
        if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient)
            return COMPUTE_DS_ONLY ? 1 : MAX_FEATURES;
        else
            return 1;
    }();

    constexpr int MEM_SIZE_NEIGHBOR_STATE = MEM_SIZE_LOCAL_STATE;

    float grad_i[MEM_SIZE_LOCAL_GRAD];
    float grad_j[MEM_SIZE_NEIGHBOR_GRAD];
    float s_i[MEM_SIZE_LOCAL_STATE];
    float s_j[MEM_SIZE_NEIGHBOR_STATE];
    float v_i, v_j; // Specific volume (mass / density)
    float m_i, m_j;
    float default_mass = config.default_mass;
    float eps2 = config.eps * config.eps;

    float grad_x[DIM] = {0.0f}; // Gradient w.r.t. position
    float grad_rho = 0.0f;      // Gradient w.r.t. density
    float grad_s[MAX_FEATURES]; // Gradient w.r.t. state
    for (uint32_t f = 0; f < feature_dim; ++f)
        grad_s[f] = 0.0f;

    int incoming_grad_buffer_dim;
    const float *incoming_grad_buffer;
    float coef;
    if constexpr (COMP_TYPE == ComputationType::Density)
    {
        incoming_grad_buffer_dim = 1;
        incoming_grad_buffer = dL_rho_out;
        coef = config.smoothing_coef;
    }
    else if constexpr (COMP_TYPE == ComputationType::Blur)
    {
        incoming_grad_buffer_dim = feature_dim;
        incoming_grad_buffer = dL_blur_state_out;
        coef = config.smoothing_coef;
    }
    else if constexpr (COMP_TYPE == ComputationType::Gradient)
    {
        incoming_grad_buffer_dim = feature_dim * DIM;
        incoming_grad_buffer = dL_grad_state_out;
        coef = config.spiky_coef;
    }
    else if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
    {
        incoming_grad_buffer_dim = DIM * DIM;
        incoming_grad_buffer = dL_moment_matrix_out;
        coef = config.spiky_coef;
    }
    else if constexpr (COMP_TYPE == ComputationType::DensityGradient)
    {
        incoming_grad_buffer_dim = DIM;
        incoming_grad_buffer = dL_grad_density_out;
        coef = config.spiky_coef;
    }

    constexpr int chunk_types = []()
    {
        if constexpr (BC == BoundaryCondition::PERIODIC)
            return 2;
        else
            return 1;
    }();
    constexpr int NUM_NEIGHBOR_CELLS = (DIM == 2) ? 9 : 27;

    // load particle position
    ParticlePosition<DIM> p_i = read_position<DIM>(position_in, config.particle_count, batch_idx, particle_idx);

    m_i = read_mass(mass_in, config.particle_count, batch_idx, particle_idx, default_mass);

    // local particle density
    if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::MomentMatrix)
        v_i = m_i / read_state(rho_in, config.particle_count, 1, batch_idx, particle_idx, 0);

    float v_i_sq_over_m_i = 0.0f;
    if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::MomentMatrix)
        v_i_sq_over_m_i = (v_i * v_i) / m_i;

    // local particle state
    if constexpr (COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::Blur)
        if constexpr (!COMPUTE_DS_ONLY)
            for (uint32_t f = 0; f < feature_dim; ++f)
                s_i[f] = read_state(state_in, config.particle_count, feature_dim, batch_idx, particle_idx, f);

    if constexpr (!COMPUTE_DS_ONLY || COMP_TYPE != ComputationType::Blur)
        for (uint32_t k = 0; k < incoming_grad_buffer_dim; ++k)
            grad_i[k] = read_state(incoming_grad_buffer, config.particle_count, incoming_grad_buffer_dim, batch_idx, particle_idx, k);

    int local_cell[DIM];
    int neighbor_cell[DIM];
    config.pos2cell(p_i, local_cell);
    uint32_t local_cell_idx = config.cell2hash(local_cell);

    // TODO: Switch to Morton Hashing for better memory access patterns
    // Change GridConfig accordingly
    for (int k = 0; k < NUM_NEIGHBOR_CELLS; k++)
    {
        if constexpr (DIM == 2)
        {
            neighbor_cell[0] = local_cell[0] + (k % 3 - 1);
            neighbor_cell[1] = local_cell[1] + (k / 3 - 1);
        }
        else if constexpr (DIM == 3)
        {
            neighbor_cell[0] = local_cell[0] + (k % 3 - 1);
            neighbor_cell[1] = local_cell[1] + ((k / 3) % 3 - 1);
            neighbor_cell[2] = local_cell[2] + (k / 9 - 1);
        }

        if (!config.is_in_bound(neighbor_cell))
            continue;

        config.apply_boundary(neighbor_cell);
        uint32_t neighbor_cell_idx = config.cell2hash(neighbor_cell);

        uint32_t remote_start = bin_offsets[batch_idx * config.cell_count + neighbor_cell_idx] - batch_idx * config.particle_count;
        uint32_t remote_end = bin_offsets[batch_idx * config.cell_count + neighbor_cell_idx + 1] - batch_idx * config.particle_count;
        uint32_t remote_count = remote_end - remote_start;
        if (remote_count == 0)
            continue;

        for (uint32_t j = 0; j < remote_count; ++j)
        {
            ParticlePosition<DIM> p_j = read_position<DIM>(position_in, config.particle_count, batch_idx, remote_start + j);
            if (config.l2_distance(p_i, p_j) >= eps2)
                continue;

            if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::MomentMatrix)
                if (!COMPUTE_DS_ONLY || COMP_TYPE != ComputationType::Blur)
                {
                    m_j = read_mass(mass_in, config.particle_count, batch_idx, remote_start + j, default_mass);
                    v_j = m_j / read_state(rho_in, config.particle_count, 1, batch_idx, remote_start + j, 0);
                }

            if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient)
                if constexpr (!COMPUTE_DS_ONLY)
                    for (uint32_t f = 0; f < feature_dim; ++f)
                        s_j[f] = read_state(state_in, config.particle_count, feature_dim, batch_idx, remote_start + j, f);

            for (uint32_t k = 0; k < incoming_grad_buffer_dim; ++k)
                grad_j[k] = read_state(incoming_grad_buffer, config.particle_count, incoming_grad_buffer_dim, batch_idx, remote_start + j, k);

            if constexpr (COMP_TYPE == ComputationType::Density)
            {
                m_j = read_mass(mass_in, config.particle_count, batch_idx, remote_start + j, default_mass);
                float grad_coef = grad_i[0] * m_j + grad_j[0] * m_i;
                ParticlePosition<DIM> r_ij = config.displacement(p_j, p_i);
                backward_smoothing_kernel_inplace(r_ij, config.eps, grad_coef, &grad_x[0]);
            }

            if constexpr (COMP_TYPE == ComputationType::Blur)
            {
                ParticlePosition<DIM> r_ij = config.displacement(p_j, p_i);
                float w_ij = smoothing_kernel(r_ij, config.eps);
                for (uint32_t f = 0; f < feature_dim; ++f)
                {
                    grad_s[f] += w_ij * v_i * grad_j[f];
                    if constexpr (!COMPUTE_DS_ONLY)
                    {
                        grad_rho += -grad_j[f] * w_ij * s_i[f] * v_i_sq_over_m_i;
                        backward_smoothing_kernel_inplace(r_ij, config.eps,
                                                          grad_i[f] * s_j[f] * v_j + grad_j[f] * s_i[f] * v_i, &grad_x[0]);
                    }
                }
            }

            if constexpr (COMP_TYPE == ComputationType::Gradient)
            {
                float w_ij[DIM] = {0.0f};
                ParticlePosition<DIM> r_ij = config.displacement(p_j, p_i);
                spiky_kernel_inplace(r_ij, config.eps, 1.0, w_ij);
                for (uint32_t f = 0; f < feature_dim; ++f)
                {
                    float grad_diff[DIM] = {0.0f};
                    for (uint32_t d = 0; d < DIM; ++d)
                    {
                        float c = grad_j[f * DIM + d] * v_i + grad_i[f * DIM + d] * v_j;
                        grad_s[f] += c * w_ij[d];
                        if constexpr (!COMPUTE_DS_ONLY)
                        {
                            grad_rho += -grad_j[f * DIM + d] * (s_i[f] - s_j[f]) * w_ij[d] * v_i_sq_over_m_i;
                            grad_diff[d] = (s_i[f] - s_j[f]) * c;
                        }
                    }
                    if constexpr (!COMPUTE_DS_ONLY)
                        backward_spiky_kernel_inplace(r_ij, config.eps, grad_diff, &grad_x[0]);
                }
            }

            if constexpr (COMP_TYPE == ComputationType::DensityGradient)
            {
                m_j = read_mass(mass_in, config.particle_count, batch_idx, remote_start + j, default_mass);
                float grad_diff[DIM];
                for (uint32_t d = 0; d < DIM; ++d)
                    grad_diff[d] = grad_j[d] * m_i - grad_i[d] * m_j;

                ParticlePosition<DIM> r_ij = config.displacement(p_j, p_i);
                backward_spiky_kernel_inplace(r_ij, config.eps, grad_diff, &grad_x[0]);
            }

            if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
            {
                ParticlePosition<DIM> r_ij = config.displacement(p_j, p_i);
                float r[DIM];
                for (uint32_t d = 0; d < DIM; ++d)
                    r[d] = r_ij[d];

                float w_ij[DIM] = {0.0f};
                spiky_kernel_inplace(r_ij, config.eps, 1.0, w_ij);

                float grad_diff[DIM * DIM];
                for (uint32_t d1 = 0; d1 < DIM; ++d1)
                    for (uint32_t d2 = 0; d2 < DIM; ++d2)
                    {
                        grad_rho += -grad_j[d1 * DIM + d2] * r[d1] * w_ij[d2] * v_i_sq_over_m_i;
                        float c = grad_i[d1 * DIM + d2] * v_j + grad_j[d1 * DIM + d2] * v_i;
                        grad_x[d1] += w_ij[d2] * c;
                        grad_diff[d1 * DIM + d2] = r[d1] * c;
                    }

                for (uint32_t d = 0; d < DIM; ++d)
                    backward_spiky_kernel_inplace(r_ij, config.eps, &grad_diff[d * DIM], &grad_x[0]);
            }
        }
    }

    if constexpr (!COMPUTE_DS_ONLY)
    {

        uint32_t idx = batch_idx * config.particle_count + particle_idx;
        if constexpr (DIM == 2)
            dL_position_in[idx].pos = make_float2(grad_x[0] * coef,
                                                  grad_x[1] * coef);
        else if constexpr (DIM == 3)
            dL_position_in[idx].pos = make_float4(grad_x[0] * coef,
                                                  grad_x[1] * coef,
                                                  grad_x[2] * coef, 0.0f);
    }

    if constexpr (!COMPUTE_DS_ONLY)
        if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::MomentMatrix)
            dL_rho_in[batch_idx * config.particle_count + particle_idx] = grad_rho * coef;

    if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient)
        for (uint32_t f = 0; f < feature_dim; ++f)
            dL_state_in[(batch_idx * config.particle_count + particle_idx) * feature_dim + f] = grad_s[f] * coef;
}

template <int DIM, ComputationType COMP_TYPE, int STRIDE = 128,
          BoundaryCondition BC = BoundaryCondition::PERIODIC,
          KernelStrategy ST = KernelStrategy::GridBased,
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
    const uint32_t * __restrict__ total_blocks, // Total number of blocks in block_info
    const uint32_t * __restrict__ bin_offsets, // Shape (batch_size * cell_count + 1)
    const BlockInfo * __restrict__ block_info, // Block information array
    const GridConfig<DIM, BC, ST> config)
{
    if (blockIdx.x >= *total_blocks)
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
    int incoming_grad_buffer_dim;
    uint32_t feature_dim_log2 = 0;
    bool is_feature_dim_pow2;
    uint32_t grad_dim_log2 = 0;
    bool is_grad_dim_pow2;
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
        is_grad_dim_pow2 = true;
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
        is_feature_dim_pow2 = (feature_dim & (feature_dim - 1)) == 0;
        while ((1U << feature_dim_log2) < feature_dim)
            feature_dim_log2++;

        is_grad_dim_pow2 = is_feature_dim_pow2;
        grad_dim_log2 = feature_dim_log2;
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
        is_feature_dim_pow2 = (feature_dim & (feature_dim - 1)) == 0;
        while ((1U << feature_dim_log2) < feature_dim)
            feature_dim_log2++;

        is_grad_dim_pow2 = (incoming_grad_buffer_dim & (incoming_grad_buffer_dim - 1)) == 0;
        while ((1U << grad_dim_log2) < incoming_grad_buffer_dim)
            grad_dim_log2++;
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

        is_grad_dim_pow2 = DIM == 2;
        grad_dim_log2 = DIM == 2 ? 2 : 3;
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
        is_grad_dim_pow2 = (DIM & (DIM - 1)) == 0;
        grad_dim_log2 = DIM == 2 ? 1 : 2;
    }
    else
    {
        return; // Unsupported computation type
    }

    BlockInfo b_info = block_info[blockIdx.x];
    int global_cell_idx = b_info.cell_idx;
    int local_cell_idx = global_cell_idx % config.cell_count; // Extract local cell index from global

    uint32_t batch_idx = b_info.batch_idx;
    constexpr int NUM_ROWS = (DIM == 2) ? 3 : 9; // Number of rows of neighbor cells to consider

    int cell[DIM];
    int neighbor_cell[DIM];
    config.hash2cell(local_cell_idx, cell);

    uint32_t local_start = bin_offsets[global_cell_idx] - batch_idx * config.particle_count;
    uint32_t local_end = bin_offsets[global_cell_idx + 1] - batch_idx * config.particle_count;
    local_start += b_info.offset_in_cell;
    local_end = min(local_end, local_start + b_info.particle_count);
    uint32_t local_count = local_end - local_start;
    if (local_count <= 0)
        return;

    float eps2 = config.eps * config.eps;

    // Outermost loop: chunk types.
    constexpr int chunk_types = []()
    {
        if constexpr (BC == BoundaryCondition::PERIODIC)
            return 2;
        else
            return 1;
    }();

    ParticlePosition<DIM> *dL_dx_global;
    float *dL_drho_global, *dL_ds_global;

    // First loop: local chunks
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
                uint32_t particle_idx = is_grad_dim_pow2 ? k >> grad_dim_log2 : k / incoming_grad_buffer_dim;
                uint32_t grad_idx = is_grad_dim_pow2 ? k & (incoming_grad_buffer_dim - 1) : k % incoming_grad_buffer_dim;
                dL_local[k] = read_state(incoming_grad_buffer,
                                         config.particle_count,
                                         incoming_grad_buffer_dim,
                                         batch_idx,
                                         local_start + local_offset + particle_idx,
                                         grad_idx);
            }
        }
        // Each thread loads local state if needed
        if constexpr (!COMPUTE_DS_ONLY)
        {
            if constexpr (COMP_TYPE == ComputationType::Gradient || COMP_TYPE == ComputationType::Blur)
            {
                for (uint32_t k = threadIdx.x; k < local_chunk * feature_dim; k += blockDim.x)
                {
                    uint32_t particle_idx = is_feature_dim_pow2 ? k >> feature_dim_log2 : k / feature_dim;
                    uint32_t feature_idx = is_feature_dim_pow2 ? k & (feature_dim - 1) : k % feature_dim;
                    local_state[k] = read_state(state_in,
                                                config.particle_count,
                                                feature_dim,
                                                batch_idx, local_start + local_offset + particle_idx,
                                                feature_idx);
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

        for (int row_idx = 0; row_idx < NUM_ROWS; row_idx++)
        {
            if constexpr (DIM == 2)
            {
                neighbor_cell[0] = cell[0];
                neighbor_cell[1] = cell[1] + (row_idx - 1);

                if (!config.is_in_bound(neighbor_cell))
                    continue;

                config.apply_boundary(neighbor_cell);
            }
            else if constexpr (DIM == 3)
            {
                neighbor_cell[0] = cell[0];
                neighbor_cell[1] = cell[1] + (row_idx / 3 - 1);
                neighbor_cell[2] = cell[2] + (row_idx % 3 - 1);
                if (!config.is_in_bound(neighbor_cell))
                    continue;
                config.apply_boundary(neighbor_cell);
            }
            for (int chunk_type = 0; chunk_type < chunk_types; chunk_type++)
            {
                // Handle wrap-around for x dimension if config.boundary_condition == PERIODIC
                int remote_x_start, remote_x_end;
                if (chunk_type == 1)
                {
                    if (cell[0] > 0 && cell[0] < config.grid_size[0] - 1)
                        continue;
                    else
                    {
                        if (cell[0] == 0)
                        {
                            remote_x_start = config.grid_size[0] - 1;
                            remote_x_end = config.grid_size[0] - 1;
                        }
                        else if (cell[0] == config.grid_size[0] - 1)
                        {
                            remote_x_start = 0;
                            remote_x_end = 0;
                        }
                    }
                }
                else
                {
                    remote_x_start = max(0, cell[0] - 1);
                    remote_x_end = min(config.grid_size[0] - 1, cell[0] + 1);
                }

                neighbor_cell[0] = remote_x_start;
                uint32_t remote_cell_start_idx = config.cell2hash(neighbor_cell);
                neighbor_cell[0] = remote_x_end;
                uint32_t remote_cell_end_idx = config.cell2hash(neighbor_cell);

                uint32_t remote_start = bin_offsets[batch_idx * config.cell_count + remote_cell_start_idx] - batch_idx * config.particle_count;
                uint32_t remote_end = bin_offsets[batch_idx * config.cell_count + remote_cell_end_idx + 1] - batch_idx * config.particle_count;

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
                                uint32_t particle_idx = is_feature_dim_pow2 ? k >> feature_dim_log2 : k / feature_dim;
                                uint32_t feature_idx = is_feature_dim_pow2 ? k & (feature_dim - 1) : k % feature_dim;
                                neighbor_state[k] = read_state(state_in,
                                                               config.particle_count,
                                                               feature_dim,
                                                               batch_idx, remote_start + remote_offset + particle_idx,
                                                               feature_idx);
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
                        uint32_t particle_idx = is_grad_dim_pow2 ? k >> grad_dim_log2 : k / incoming_grad_buffer_dim;
                        uint32_t grad_idx = is_grad_dim_pow2 ? k & (incoming_grad_buffer_dim - 1) : k % incoming_grad_buffer_dim;
                        dL_neighbor[k] = read_state(incoming_grad_buffer,
                                                    config.particle_count,
                                                    incoming_grad_buffer_dim,
                                                    batch_idx, remote_start + remote_offset + particle_idx,
                                                    grad_idx);
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
                            // uint32_t particle_idx = i / feature_dim;
                            // uint32_t feature_idx = i % feature_dim;
                            uint32_t particle_idx = is_feature_dim_pow2 ? i >> feature_dim_log2 : i / feature_dim;
                            uint32_t feature_idx = is_feature_dim_pow2 ? i & (feature_dim - 1) : i % feature_dim;
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
                            // uint32_t particle_idx = i / feature_dim;
                            // uint32_t feature_idx = i % feature_dim;
                            uint32_t particle_idx = is_feature_dim_pow2 ? i >> feature_dim_log2 : i / feature_dim;
                            uint32_t feature_idx = is_feature_dim_pow2 ? i & (feature_dim - 1) : i % feature_dim;
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
                    dL_dx_global[k].pos = make_float4(dL_dx_buffer[k * DIM + 0],
                                                      dL_dx_buffer[k * DIM + 1],
                                                      dL_dx_buffer[k * DIM + 2], 0.0f);
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
    }
}

template <int DIM, ComputationType COMP_TYPE, int STRIDE = 128, BoundaryCondition BC = BoundaryCondition::PERIODIC, KernelStrategy ST = KernelStrategy::GridBased>
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
    const uint32_t * __restrict__ total_blocks, // Total number of blocks in block_info
    const uint32_t * __restrict__ bin_offsets, // Shape (batch_size * cell_count + 1)
    const BlockInfo * __restrict__ block_info, // Block information array
    const GridConfig<DIM, BC, ST> config)
{

    if (blockIdx.x >= *total_blocks)
        return;

    constexpr int NUM_ROWS = (DIM == 2) ? 3 : 9;
    uint32_t output_dim = 1;
    __shared__ ParticlePosition<DIM> local_pos[STRIDE];
    __shared__ ParticlePosition<DIM> neighbor_pos[STRIDE];
    extern __shared__ float smem[];
    float *local_buffer, *neighbor_state, *local_state, *neighbor_density_rcp, *neighbor_mass, *output_buffer;
    local_buffer = smem;
    uint32_t feature_dim_log2 = 0;
    float default_mass = config.default_mass;
    bool is_pow2;
    if constexpr (COMP_TYPE == ComputationType::Blur)
    {
        neighbor_state = &local_buffer[feature_dim * STRIDE];
        neighbor_density_rcp = &neighbor_state[feature_dim * STRIDE];
        neighbor_mass = &neighbor_density_rcp[STRIDE];
        output_buffer = blur_state_out;
        output_dim = feature_dim;
        is_pow2 = (feature_dim & (feature_dim - 1)) == 0;
        while ((1U << feature_dim_log2) < feature_dim)
            feature_dim_log2++;
    }
    else if constexpr (COMP_TYPE == ComputationType::Gradient)
    {
        neighbor_state = &local_buffer[feature_dim * DIM * STRIDE];
        local_state = &neighbor_state[feature_dim * STRIDE];
        neighbor_density_rcp = &local_state[feature_dim * STRIDE];
        neighbor_mass = &neighbor_density_rcp[STRIDE];
        output_buffer = grad_state_out;
        output_dim = feature_dim * DIM;
        is_pow2 = (feature_dim & (feature_dim - 1)) == 0;
        while ((1U << feature_dim_log2) < feature_dim)
            feature_dim_log2++;
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
    else if constexpr (COMP_TYPE == ComputationType::Count)
    {
        output_buffer = count_out;
        neighbor_mass = nullptr;
    }
    else
        output_buffer = nullptr;

    // BlockInfo b_info = block_info[batch_idx * config.max_blocks_per_batch + blockIdx.x];
    BlockInfo b_info = block_info[blockIdx.x];
    uint32_t batch_idx = b_info.batch_idx;

    if (b_info.particle_count == 0)
        return;

    if (batch_idx >= config.batch_size)
        return;

    int global_cell_idx = b_info.cell_idx;
    int local_cell_idx = global_cell_idx % config.cell_count; // Extract local cell index from global

    int cell[DIM];
    int neighbor_cell[DIM];
    config.hash2cell(local_cell_idx, cell);

    uint32_t local_start = bin_offsets[global_cell_idx] - batch_idx * config.particle_count;
    uint32_t local_end = bin_offsets[global_cell_idx + 1] - batch_idx * config.particle_count;
    local_start += b_info.offset_in_cell;
    local_end = min(local_end, local_start + b_info.particle_count);
    uint32_t local_count = local_end - local_start;
    if (local_count <= 0)
        return;

    float eps2 = config.eps * config.eps;

    // Outermost loop: chunk types.
    constexpr int chunk_types = []()
    {
        if constexpr (BC == BoundaryCondition::PERIODIC)
            return 2;
        else
            return 1;
    }();

    // First loop: local chunks
    for (uint32_t local_offset = 0; local_offset < local_count; local_offset += STRIDE)
    {
        uint32_t local_chunk = min(STRIDE, local_count - local_offset);

        // Each thread loads a few particles into shared memory
        for (uint32_t k = threadIdx.x; k < local_chunk; k += blockDim.x)
            local_pos[k] = read_position<DIM>(position_in, config.particle_count, batch_idx,
                                              local_start + local_offset + k);

        // Each thread loads local state if needed
        if constexpr (COMP_TYPE == ComputationType::Gradient)
        {
            for (uint32_t k = threadIdx.x; k < local_chunk * feature_dim; k += blockDim.x)
            {
                int particle_idx = is_pow2 ? k >> feature_dim_log2 : k / feature_dim;
                int feature_idx = is_pow2 ? k & (feature_dim - 1) : k % feature_dim;
                local_state[k] = read_state(state_in, config.particle_count, feature_dim, batch_idx, local_start + local_offset + particle_idx, feature_idx);
            }
        }

        float *output_buffer_global = output_buffer + (batch_idx * config.particle_count + local_start + local_offset) * output_dim;

        for (uint32_t k = threadIdx.x; k < local_chunk * output_dim; k += blockDim.x)
            local_buffer[k] = 0.0f;

        for (int row_idx = 0; row_idx < NUM_ROWS; row_idx++)
        {
            if constexpr (DIM == 2)
            {
                neighbor_cell[0] = cell[0];
                neighbor_cell[1] = cell[1] + (row_idx - 1);

                if (!config.is_in_bound(neighbor_cell))
                    continue;

                config.apply_boundary(neighbor_cell);
            }
            else if constexpr (DIM == 3)
            {
                neighbor_cell[0] = cell[0];
                neighbor_cell[1] = cell[1] + (row_idx / 3 - 1);
                neighbor_cell[2] = cell[2] + (row_idx % 3 - 1);
                if (!config.is_in_bound(neighbor_cell))
                    continue;
                config.apply_boundary(neighbor_cell);
            }
            for (int chunk_type = 0; chunk_type < chunk_types; chunk_type++)
            {
                // Handle wrap-around for x dimension if config.boundary_condition == PERIODIC
                int remote_x_start, remote_x_end;
                if (chunk_type == 1)
                {
                    if (cell[0] > 0 && cell[0] < config.grid_size[0] - 1)
                        continue;
                    else
                    {
                        if (cell[0] == 0)
                        {
                            remote_x_start = config.grid_size[0] - 1;
                            remote_x_end = config.grid_size[0] - 1;
                        }
                        else if (cell[0] == config.grid_size[0] - 1)
                        {
                            remote_x_start = 0;
                            remote_x_end = 0;
                        }
                    }
                }
                else
                {
                    remote_x_start = max(0, cell[0] - 1);
                    remote_x_end = min(config.grid_size[0] - 1, cell[0] + 1);
                }

                neighbor_cell[0] = remote_x_start;
                uint32_t remote_cell_start_idx = config.cell2hash(neighbor_cell);
                neighbor_cell[0] = remote_x_end;
                uint32_t remote_cell_end_idx = config.cell2hash(neighbor_cell);

                uint32_t remote_start = bin_offsets[batch_idx * config.cell_count + remote_cell_start_idx] - batch_idx * config.particle_count;
                uint32_t remote_end = bin_offsets[batch_idx * config.cell_count + remote_cell_end_idx + 1] - batch_idx * config.particle_count;

                uint32_t remote_count = remote_end - remote_start;
                if (remote_count == 0)
                    continue;

                for (uint32_t remote_offset = 0; remote_offset < remote_count; remote_offset += STRIDE)
                {
                    uint32_t remote_chunk = min(STRIDE, remote_count - remote_offset);
                    // Each thread loads remote particle positions into shared memory
                    for (uint32_t k = threadIdx.x; k < remote_chunk; k += blockDim.x)
                        neighbor_pos[k] = read_position<DIM>(position_in,
                                                             config.particle_count, batch_idx,
                                                             remote_start + remote_offset + k);

                    // Each thread loads remote particle mass/density if needed
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

                    // Better implementation (coalesced access)
                    if constexpr (COMP_TYPE == ComputationType::Blur || COMP_TYPE == ComputationType::Gradient)
                    {
                        for (uint32_t k = threadIdx.x; k < remote_chunk * feature_dim; k += blockDim.x)
                        {
                            int particle_idx = is_pow2 ? k >> feature_dim_log2 : k / feature_dim;
                            int feature_idx = is_pow2 ? k & (feature_dim - 1) : k % feature_dim;
                            neighbor_state[k] = read_state(state_in, config.particle_count, feature_dim, batch_idx, remote_start + remote_offset + particle_idx, feature_idx);
                        }
                    }
                    __syncthreads();

                    // Nested for loops to compute local/remote particles interactions
                    if constexpr (COMP_TYPE == ComputationType::Count || COMP_TYPE == ComputationType::Density)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk; i += blockDim.x)
                        {
                            ParticlePosition<DIM> p_i = local_pos[i];
                            float output = 0.0f;

                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                // TODO: Think of offset + mod for improved shared memory access pattern (less bank conflicts)
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
                            // atomicAdd(&output_buffer_global[i], output * config.smoothing_coef);

                            if constexpr (COMP_TYPE == ComputationType::Count)
                                local_buffer[i] += output;
                            // atomicAdd(&output_buffer_global[i], output);
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::Blur)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk * output_dim; i += blockDim.x)
                        {
                            uint32_t particle_idx = is_pow2 ? i >> feature_dim_log2 : i / feature_dim;
                            uint32_t feature_idx = is_pow2 ? i & (feature_dim - 1) : i % feature_dim;
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
                            // atomicAdd(&output_buffer_global[i], output * config.smoothing_coef);
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::Gradient)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk * feature_dim; i += blockDim.x)
                        // for (uint32_t i = threadIdx.x; i < local_chunk * output_dim; i += blockDim.x)
                        {
                            uint32_t particle_idx = is_pow2 ? i >> feature_dim_log2 : i / feature_dim;
                            uint32_t feature_idx = is_pow2 ? i & (feature_dim - 1) : i % feature_dim;
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
                                    // float gradw_ij = gradient_kernel(p_j - p_i, config.eps, dim_idx);
                                    // output += (s_j - s_i) * gradw_ij / rho_j;
                                    ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                                    spiky_kernel_inplace(r_ij, config.eps, (s_j - s_i) * v_j, output);
                                }
                            }
                            // local_buffer[i] += output * config.spiky_coef;
                            for (uint32_t dim = 0; dim < DIM; ++dim)
                                local_buffer[i * DIM + dim] += output[dim] * config.spiky_coef;
                            // atomicAdd(&output_buffer_global[i * DIM + dim], output[dim] * config.spiky_coef);
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::DensityGradient)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk; i += blockDim.x)
                        {
                            uint32_t particle_idx = i; // local particle index in chunk to process
                            ParticlePosition<DIM> p_i = local_pos[particle_idx];
                            float output[DIM] = {0.0f};

                            for (uint32_t j = 0; j < remote_chunk; ++j)
                            {
                                ParticlePosition<DIM> p_j = neighbor_pos[j];

                                if (config.l2_distance(p_i, p_j) < eps2)
                                {
                                    // output += gradient_kernel(p_j - p_i, config.eps, dim_idx);
                                    ParticlePosition<DIM> r_ij = config.displacement(p_i, p_j);
                                    spiky_kernel_inplace(r_ij, config.eps, neighbor_mass[j], output);
                                }
                            }
                            // local_buffer[i] += output * config.spiky_coef;
                            for (uint32_t dim = 0; dim < DIM; ++dim)
                                local_buffer[i * DIM + dim] += output[dim] * config.spiky_coef;
                            // atomicAdd(&output_buffer_global[i], output * config.smoothing_coef);
                        }
                    }
                    if constexpr (COMP_TYPE == ComputationType::MomentMatrix)
                    {
                        for (uint32_t i = threadIdx.x; i < local_chunk; i += blockDim.x)
                        {
                            uint32_t particle_idx = i; // local particle index in chunk to process
                            ParticlePosition<DIM> p_i = local_pos[particle_idx];
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
                            // atomicAdd(&output_buffer_global[i * DIM * DIM + m], output[m]);
                        }
                    }
                    __syncthreads();
                }
            }
        }

        for (uint32_t k = threadIdx.x; k < local_chunk * output_dim; k += blockDim.x)
        {
            output_buffer_global[k] = local_buffer[k];
        }
    }
}

template <int DIM, BoundaryCondition BC, KernelStrategy ST>
class HashGrid
{
    static_assert(DIM == 2 || DIM == 3, "HashGrid only supports 2D or 3D");

protected:
    GridConfig<DIM, BC, ST> m_config;

    uint32_t *m_cell_counts = nullptr;       // Device array of size (cell_count * batch_size)
    uint32_t *m_bin_write_indices = nullptr; // Device array of size (cell_count * batch_size)

    // Pre-allocated temporary storage for CUB prefix sum
    void *m_temp_storage = nullptr;
    size_t m_temp_storage_bytes = 0;

    // BlockInfo *m_block_info = nullptr;     // Device array storing block information
    uint32_t *m_blocks_per_cell = nullptr; // Device array storing blocks needed per cell (batch_size * cell_count)
    uint32_t *m_block_offsets = nullptr;   // Device array for cumulative sum of blocks per cell (batch_size * cell_count + 1)

    uint32_t m_dimension;  // 2 for 2D, 4 for 3D (to align with float4)
    uint32_t m_max_blocks; // Number of blocks used in the last computation

    void _init(float eps, uint32_t particle_count, uint32_t batch_size, uint32_t max_particles_per_block)
    {
        m_config.eps = eps;
        m_config.inv_eps = 1.0f / eps;
        m_config.particle_count = particle_count;
        m_config.batch_size = batch_size;
        m_config.MAX_PARTICLES_PER_BLOCK = max_particles_per_block;
        m_config.default_mass = 1.0f;
        for (uint32_t d = 0; d < DIM; ++d)
        {
            m_config.L[d] = static_cast<float>(m_config.grid_size[d]) * m_config.eps;
            m_config.half_L[d] = 0.5f * m_config.L[d];
        }
        if constexpr (ST == KernelStrategy::GridBased)
        {
            m_config.max_blocks_per_batch = min(particle_count,
                                                m_config.cell_count + (particle_count + m_config.MAX_PARTICLES_PER_BLOCK - 1) / m_config.MAX_PARTICLES_PER_BLOCK);
            m_max_blocks = m_config.max_blocks_per_batch * batch_size;
        }

        else
        {
            m_config.max_blocks_per_batch = 0; // dummy, not used
            m_max_blocks = 1;                  // dummy, not used. Set to 1 to avoid zero-sized allocations.
        }

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
        cuda_check(cudaMalloc(&m_cell_counts, sizeof(uint32_t) * m_config.cell_count * batch_size));
        cuda_check(cudaMalloc(&m_bin_write_indices, sizeof(uint32_t) * m_config.cell_count * batch_size));

        // Determine temporary storage requirements for CUB prefix sum
        cub::DeviceScan::ExclusiveSum(
            m_temp_storage, m_temp_storage_bytes,
            m_cell_counts, (uint32_t *)nullptr, batch_size * m_config.cell_count + 1);

        cuda_check(cudaMalloc(&m_temp_storage, m_temp_storage_bytes));

        if constexpr (ST == KernelStrategy::GridBased)
        {
            // Allocate blocks per cell array
            cuda_check(cudaMalloc(&m_blocks_per_cell, sizeof(uint32_t) * m_config.cell_count * batch_size));
            cuda_check(cudaMalloc(&m_block_offsets, sizeof(uint32_t) * (m_config.cell_count * batch_size + 1)));
        }

        m_dimension = (DIM == 2) ? 2U : 4U;
    }

public:
    ~HashGrid()
    {
        cudaFree(m_cell_counts);
        cudaFree(m_bin_write_indices);
        cudaFree(m_temp_storage);

        if constexpr (ST == KernelStrategy::GridBased)
            cudaFree(m_blocks_per_cell);

        if constexpr (ST == KernelStrategy::GridBased)
            cudaFree(m_block_offsets);
    }

    float eps() const { return m_config.eps; }
    uint32_t particle_count() const { return m_config.particle_count; }
    uint32_t batch_size() const { return m_config.batch_size; }

    int grid_size_x() const { return m_config.grid_size[0]; }
    int grid_size_y() const { return m_config.grid_size[1]; }

    int max_blocks() const { return m_max_blocks; }

    uint32_t max_particles_per_block() const { return m_config.MAX_PARTICLES_PER_BLOCK; }

    void initialize(
        FloatArray3D positions,   // Shape (batch_size, N, DIM)
        IntArray2D permutation,   // Shape (batch_size, N)
        IntArray1D bin_offsets,   // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info // Shape (max_blocks, 4)
    )
    {
        validate_array_shape(positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "positions");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(permutation, {m_config.batch_size, m_config.particle_count}, "permutation");

        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_positions = reinterpret_cast<const ParticlePosition<DIM> *>(positions.data());

        cuda_check(cudaMemset(m_cell_counts, 0, sizeof(int32_t) * (m_config.cell_count * m_config.batch_size)));
        if constexpr (ST == KernelStrategy::GridBased)
            cuda_check(cudaMemset(m_blocks_per_cell, 0, sizeof(uint32_t) * (m_config.cell_count * m_config.batch_size)));

        // Launch kernel
        dim3 block_dim(256, 1, 1);
        dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
        grid_count_particles<DIM, BC, ST><<<grid_dim, block_dim>>>(
            d_positions, m_cell_counts, m_blocks_per_cell, m_config);
        cuda_check(cudaGetLastError());

        uint32_t *d_permutation = reinterpret_cast<uint32_t *>(permutation.data());
        uint32_t *d_bin_offsets = reinterpret_cast<uint32_t *>(bin_offsets.data());

        cuda_check(cudaMemset(d_permutation, 0, sizeof(uint32_t) * (m_config.particle_count * m_config.batch_size)));
        cuda_check(cudaMemset(d_bin_offsets, 0, sizeof(uint32_t) * (m_config.batch_size * m_config.cell_count + 1)));
        cuda_check(cudaMemset(m_bin_write_indices, 0, sizeof(uint32_t) * (m_config.cell_count * m_config.batch_size)));

        // Compute prefix sum to get bin offsets
        cub::DeviceScan::ExclusiveSum(
            m_temp_storage,
            m_temp_storage_bytes,
            m_cell_counts,
            d_bin_offsets,
            m_config.batch_size * m_config.cell_count + 1);

        // Launch kernel to compute permutation
        grid_compute_permutation<<<grid_dim, block_dim>>>(
            d_positions, d_bin_offsets, m_bin_write_indices, d_permutation, m_config);

        if constexpr (ST == KernelStrategy::GridBased)
        {
            BlockInfo *d_block_info = reinterpret_cast<BlockInfo *>(block_info.data());
            // Compute prefix sum to get block offsets
            cuda_check(cudaMemset(m_block_offsets, 0, sizeof(uint32_t) * (m_config.cell_count * m_config.batch_size + 1)));
            cub::DeviceScan::ExclusiveSum(
                m_temp_storage,
                m_temp_storage_bytes,
                m_blocks_per_cell,
                m_block_offsets,
                m_config.batch_size * m_config.cell_count + 1);

            cuda_check(cudaMemset(d_block_info, 0, sizeof(BlockInfo) * m_max_blocks));
            // Launch kernel to assign block information
            block_dim = dim3(256, 1, 1);
            grid_dim = dim3((m_config.cell_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
            assign_block_info<<<grid_dim, block_dim>>>(
                m_cell_counts, m_block_offsets, d_block_info, m_config);
        }
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
        validate_array_shape(input_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "input_positions");
        validate_array_shape(input_state, {m_config.batch_size, m_config.particle_count, feature_dim}, "input_state");
        validate_array_shape(input_mass, {m_config.batch_size, m_config.particle_count}, "input_mass");
        validate_array_shape(permutation, {m_config.batch_size, m_config.particle_count}, "permutation");
        validate_array_shape(output_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "output_positions");
        validate_array_shape(output_state, {m_config.batch_size, m_config.particle_count, feature_dim}, "output_state");
        validate_array_shape(output_mass, {m_config.batch_size, m_config.particle_count}, "output_mass");

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
        validate_array_shape(grad_output_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "grad_output_positions");
        validate_array_shape(grad_output_state, {m_config.batch_size, m_config.particle_count, feature_dim}, "grad_output_state");
        validate_array_shape(grad_output_mass, {m_config.batch_size, m_config.particle_count}, "grad_output_mass");
        validate_array_shape(permutation, {m_config.batch_size, m_config.particle_count}, "permutation");
        validate_array_shape(grad_input_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "grad_input_positions");
        validate_array_shape(grad_input_state, {m_config.batch_size, m_config.particle_count, feature_dim}, "grad_input_state");
        validate_array_shape(grad_input_mass, {m_config.batch_size, m_config.particle_count}, "grad_input_mass");

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

    void density_forward(
        FloatArray3D binned_positions, // Shape (batch_size, N, DIM)
        FloatArray2D mass,             // Input mass, Shape (batch_size, N)
        FloatArray2D rho,              // Output density, Shape (batch_size, N)
        IntArray1D bin_offsets,        // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info      // Output block info, Shape (max_blocks, 4)
    )
    {
        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(mass, {m_config.batch_size, m_config.particle_count}, "mass");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(rho, {m_config.batch_size, m_config.particle_count}, "rho");
        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        const float *d_mass = mass.data();

        if constexpr (ST == KernelStrategy::GridBased)
        {
            const BlockInfo *d_block_info = reinterpret_cast<const BlockInfo *>(block_info.data());
            dim3 block_dim(128, 1, 1);
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 grid_dim(m_max_blocks, 1, 1);

            // determine shared memory size
            const int STRIDE = 128;
            size_t smem_size = STRIDE * 2 * sizeof(float); // local_buffer + neighbor_mass

            forward_kernel<DIM, ComputationType::Density, STRIDE><<<grid_dim, block_dim, smem_size>>>(
                d_binned_positions, // Input binned positions
                nullptr,            // No input density
                d_mass,             // Input mass
                nullptr,            // No input state
                rho.data(),         // Output density
                nullptr,            // No count output
                nullptr,            // No blur state output
                nullptr,            // No grad state output
                nullptr,            // No grad density output
                nullptr,            // No moment matrix output
                0,                  // No feature dim
                total_blocks,       // Total blocks used
                d_bin_offsets,      // Bin offsets
                d_block_info,       // Block info
                m_config);
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
            vanilla_forward_kernel<DIM, ComputationType::Density, 0><<<grid_dim, block_dim>>>(
                d_binned_positions, // Input binned positions
                nullptr,            // No input density
                d_mass,             // Input mass
                nullptr,            // No input state
                rho.data(),         // Output density
                nullptr,            // No count output
                nullptr,            // No blur state output
                nullptr,            // No grad state output
                nullptr,            // No grad density output
                nullptr,            // No moment matrix output
                0,                  // No feature dim
                d_bin_offsets,      // Bin offsets
                m_config);
        }
    }

    void count_forward(
        FloatArray3D binned_positions, // Shape (batch_size, N, DIM)
        FloatArray2D count,            // Output count, Shape (batch_size, N)
        IntArray1D bin_offsets,        // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info      // Output block info, Shape (max_blocks, 4)
    )
    {
        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(count, {m_config.batch_size, m_config.particle_count}, "count");
        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());

        if constexpr (ST == KernelStrategy::GridBased)
        {
            const BlockInfo *d_block_info = reinterpret_cast<const BlockInfo *>(block_info.data());
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 block_dim(128, 1, 1);
            dim3 grid_dim(m_max_blocks, 1, 1);

            const int STRIDE = 128;
            size_t smem_size = STRIDE * sizeof(float); // for local_buffer
            forward_kernel<DIM, ComputationType::Count, STRIDE><<<grid_dim, block_dim, smem_size>>>(
                d_binned_positions, // Input binned positions
                nullptr,            // No input density
                nullptr,            // No input mass
                nullptr,            // No input state
                nullptr,            // No output density
                count.data(),       // Output count
                nullptr,            // No blur state output
                nullptr,            // No grad state output
                nullptr,            // No grad density output
                nullptr,            // No moment matrix output
                0,                  // No feature dim
                total_blocks,       // Total blocks used
                d_bin_offsets,      // Bin offsets
                d_block_info,       // Block info
                m_config);
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
            vanilla_forward_kernel<DIM, ComputationType::Count, 0><<<grid_dim, block_dim>>>(
                d_binned_positions, // Input binned positions
                nullptr,            // No input density
                nullptr,            // No input mass
                nullptr,            // No input state
                nullptr,            // No output density
                count.data(),       // Output count
                nullptr,            // No blur state output
                nullptr,            // No grad state output
                nullptr,            // No grad density output
                nullptr,            // No moment matrix output
                0,                  // No feature dim
                d_bin_offsets,      // Bin offsets
                m_config);
        }
    }

    void blur_forward(
        FloatArray3D binned_positions, // Shape (batch_size, N, DIM)
        FloatArray2D rho,              // Input density, Shape (batch_size, N)
        FloatArray2D mass,             // Input mass, Shape (batch_size, N)
        FloatArray3D state_in,         // Input state, Shape (batch_size, N, feature_dim)
        FloatArray3D state_out,        // Output state, Shape (batch_size, N, feature_dim)
        IntArray1D bin_offsets,        // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info      // Output block info, Shape (max_blocks, 4)
    )
    {
        uint32_t feature_dim = state_in.shape(2);
        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(rho, {m_config.batch_size, m_config.particle_count}, "rho");
        validate_array_shape(mass, {m_config.batch_size, m_config.particle_count}, "mass");
        validate_array_shape(state_in, {m_config.batch_size, m_config.particle_count, feature_dim}, "state_in");
        validate_array_shape(state_out, {m_config.batch_size, m_config.particle_count, feature_dim}, "state_out");
        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        const float *d_mass = mass.data();

        if constexpr (ST == KernelStrategy::GridBased)
        {
            // Cast block info
            const BlockInfo *d_block_info = reinterpret_cast<const BlockInfo *>(block_info.data());
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim(m_max_blocks, 1, 1);
            const int STRIDE = 64;
            size_t smem_size = STRIDE * (feature_dim + feature_dim + 2) * sizeof(float); // + neighbor_mass
            forward_kernel<DIM, ComputationType::Blur, STRIDE><<<grid_dim, block_dim, smem_size>>>(
                d_binned_positions, // Input binned positions
                rho.data(),         // Input density
                d_mass,             // Input mass
                state_in.data(),    // Input state
                nullptr,            // No output density
                nullptr,            // No count output
                state_out.data(),   // Output blur state
                nullptr,            // No grad state output
                nullptr,            // No grad density output
                nullptr,            // No moment matrix output
                feature_dim,        // Feature dimension
                total_blocks,       // Total blocks used
                d_bin_offsets,      // Bin offsets
                d_block_info,       // Block info
                m_config);
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);

            auto launch_blur_kernel = [&](auto max_features)
            {
                vanilla_forward_kernel<DIM, ComputationType::Blur, max_features><<<grid_dim, block_dim>>>(
                    d_binned_positions, // Input binned positions
                    rho.data(),         // Input density
                    d_mass,             // Input mass
                    state_in.data(),    // Input state
                    nullptr,            // No output density
                    nullptr,            // No count output
                    state_out.data(),   // Output blur state
                    nullptr,            // No grad state output
                    nullptr,            // No grad density output
                    nullptr,            // No moment matrix output
                    feature_dim,        // Feature dimension
                    d_bin_offsets,      // Bin offsets
                    m_config);
            };
            if (feature_dim <= 8)
                launch_blur_kernel(std::integral_constant<int, 8>{});
            else if (feature_dim <= 16)
                launch_blur_kernel(std::integral_constant<int, 16>{});
            else if (feature_dim <= 32)
                launch_blur_kernel(std::integral_constant<int, 32>{});
            else if (feature_dim <= 48)
                launch_blur_kernel(std::integral_constant<int, 48>{});
            else if (feature_dim <= 64)
                launch_blur_kernel(std::integral_constant<int, 64>{});
            else
                throw std::runtime_error("feature_dim exceeds maximum supported value of 32.");
        }
    }

    void gradient_forward(
        FloatArray3D binned_positions,   // Shape (batch_size, N, DIM)
        FloatArray2D rho,                // Input density, Shape (batch_size, N)
        FloatArray2D mass,               // Input mass, Shape (batch_size, N)
        FloatArray3D state_in,           // Input state, Shape (batch_size, N, feature_dim)
        FloatArray4D state_gradient_out, // Output state gradient, Shape (batch_size, N, feature_dim, DIM)
        IntArray1D bin_offsets,          // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info        // Output block info, Shape (max_blocks, 4)
    )
    {
        uint32_t feature_dim = state_in.shape(2);
        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(rho, {m_config.batch_size, m_config.particle_count}, "rho");
        validate_array_shape(mass, {m_config.batch_size, m_config.particle_count}, "mass");
        validate_array_shape(state_in, {m_config.batch_size, m_config.particle_count, feature_dim}, "state_in");
        validate_array_shape(state_gradient_out, {m_config.batch_size, m_config.particle_count, feature_dim, DIM}, "state_gradient_out");
        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        const float *d_mass = mass.data();

        if constexpr (ST == KernelStrategy::GridBased)
        {
            const BlockInfo *d_block_info = reinterpret_cast<const BlockInfo *>(block_info.data());
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim(m_max_blocks, 1, 1);
            const int STRIDE = 64;
            size_t smem_size = STRIDE * (feature_dim * DIM + feature_dim + feature_dim + 2) * sizeof(float); // + neighbor_mass

            forward_kernel<DIM, ComputationType::Gradient, STRIDE><<<grid_dim, block_dim, smem_size>>>(
                d_binned_positions,        // Input binned positions
                rho.data(),                // Input density
                d_mass,                    // Input mass
                state_in.data(),           // Input state
                nullptr,                   // No output density
                nullptr,                   // No count output
                nullptr,                   // No blur state output
                state_gradient_out.data(), // Output grad state
                nullptr,                   // No grad density output
                nullptr,                   // No moment matrix output
                feature_dim,               // Feature dimension
                total_blocks,              // Total blocks used
                d_bin_offsets,             // Bin offsets
                d_block_info,              // Block info
                m_config);
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);

            auto launch_gradient_kernel = [&](auto max_features)
            {
                vanilla_forward_kernel<DIM, ComputationType::Gradient, max_features><<<grid_dim, block_dim>>>(
                    d_binned_positions,        // Input binned positions
                    rho.data(),                // Input density
                    d_mass,                    // Input mass
                    state_in.data(),           // Input state
                    nullptr,                   // No output density
                    nullptr,                   // No count output
                    nullptr,                   // No blur state output
                    state_gradient_out.data(), // Output grad state
                    nullptr,                   // No grad density output
                    nullptr,                   // No moment matrix output
                    feature_dim,               // Feature dimension
                    d_bin_offsets,             // Bin offsets
                    m_config);
            };
            if (feature_dim <= 8)
                launch_gradient_kernel(std::integral_constant<int, 8>{});
            else if (feature_dim <= 16)
                launch_gradient_kernel(std::integral_constant<int, 16>{});
            else if (feature_dim <= 32)
                launch_gradient_kernel(std::integral_constant<int, 32>{});
            else if (feature_dim <= 48)
                launch_gradient_kernel(std::integral_constant<int, 48>{});
            else if (feature_dim <= 64)
                launch_gradient_kernel(std::integral_constant<int, 64>{});
            else
                throw std::runtime_error("feature_dim exceeds maximum supported value of 32.");
        }
    }

    void density_gradient_forward(
        FloatArray3D binned_positions,     // Shape (batch_size, N, DIM)
        FloatArray2D mass,                 // Input mass, Shape (batch_size, N)
        FloatArray3D density_gradient_out, // Output density gradient, Shape (batch_size, N, DIM)
        IntArray1D bin_offsets,            // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info          // Output block info, Shape (max_blocks, 4)
    )
    {
        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(mass, {m_config.batch_size, m_config.particle_count}, "mass");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(density_gradient_out, {m_config.batch_size, m_config.particle_count, DIM}, "density_gradient_out");
        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        const float *d_mass = mass.data();

        if constexpr (ST == KernelStrategy::GridBased)
        {
            const BlockInfo *d_block_info = reinterpret_cast<const BlockInfo *>(block_info.data());
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 block_dim(128, 1, 1);
            dim3 grid_dim(m_max_blocks, 1, 1);
            const int STRIDE = 128;
            size_t smem_size = STRIDE * (DIM + 1) * sizeof(float); // + neighbor_mass
            forward_kernel<DIM, ComputationType::DensityGradient, STRIDE><<<grid_dim, block_dim, smem_size>>>(
                d_binned_positions,          // Input binned positions
                nullptr,                     // No input density
                d_mass,                      // Input mass
                nullptr,                     // No input state
                nullptr,                     // No output density
                nullptr,                     // No count output
                nullptr,                     // No blur state output
                nullptr,                     // No grad state output
                density_gradient_out.data(), // Output density gradient
                nullptr,                     // No moment matrix output
                0,                           // No feature dim
                total_blocks,                // Total blocks used
                d_bin_offsets,               // Bin offsets
                d_block_info,                // Block info
                m_config);
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
            vanilla_forward_kernel<DIM, ComputationType::DensityGradient, 0><<<grid_dim, block_dim>>>(
                d_binned_positions,          // Input binned positions
                nullptr,                     // No input density
                d_mass,                      // Input mass
                nullptr,                     // No input state
                nullptr,                     // No output density
                nullptr,                     // No count output
                nullptr,                     // No blur state output
                nullptr,                     // No grad state output
                density_gradient_out.data(), // Output density gradient
                nullptr,                     // No moment matrix output
                0,                           // No feature dim
                d_bin_offsets,               // Bin offsets
                m_config);
        }
    }

    void moment_matrix_forward(
        FloatArray3D binned_positions,  // Shape (batch_size, N, DIM)
        FloatArray2D rho,               // Input density, Shape (batch_size, N)
        FloatArray2D mass,              // Input mass, Shape (batch_size, N)
        FloatArray4D moment_matrix_out, // Output moment matrix, Shape (batch_size, N, DIM, DIM)
        IntArray1D bin_offsets,         // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info       // Output block info, Shape (max_blocks, 4)
    )
    {
        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(rho, {m_config.batch_size, m_config.particle_count}, "rho");
        validate_array_shape(mass, {m_config.batch_size, m_config.particle_count}, "mass");
        validate_array_shape(moment_matrix_out, {m_config.batch_size, m_config.particle_count, DIM, DIM}, "moment_matrix_out");
        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        const float *d_mass = mass.data();

        if constexpr (ST == KernelStrategy::GridBased)
        {
            const BlockInfo *d_block_info = reinterpret_cast<const BlockInfo *>(block_info.data());
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 block_dim(128, 1, 1);
            dim3 grid_dim(m_max_blocks, 1, 1);
            const int STRIDE = 128;
            size_t smem_size = STRIDE * (DIM * DIM + 2) * sizeof(float); // + neighbor_mass
            forward_kernel<DIM, ComputationType::MomentMatrix, STRIDE><<<grid_dim, block_dim, smem_size>>>(
                d_binned_positions,       // Input binned positions
                rho.data(),               // Input density
                d_mass,                   // Input mass
                nullptr,                  // No input state
                nullptr,                  // No output density
                nullptr,                  // No count output
                nullptr,                  // No blur state output
                nullptr,                  // No grad state output
                nullptr,                  // No grad density output
                moment_matrix_out.data(), // Output moment matrix
                0,                        // No feature dim
                total_blocks,             // Total blocks used
                d_bin_offsets,            // Bin offsets
                d_block_info,             // Block info
                m_config);
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
            vanilla_forward_kernel<DIM, ComputationType::MomentMatrix, 0><<<grid_dim, block_dim>>>(
                d_binned_positions,       // Input binned positions
                rho.data(),               // Input density
                d_mass,                   // Input mass
                nullptr,                  // No input state
                nullptr,                  // No output density
                nullptr,                  // No count output
                nullptr,                  // No blur state output
                nullptr,                  // No grad state output
                nullptr,                  // No grad density output
                moment_matrix_out.data(), // Output moment matrix
                0,                        // No feature dim
                d_bin_offsets,            // Bin offsets
                m_config);
        }
    }

    void density_backward(
        FloatArray3D binned_positions, // Shape (batch_size, N, DIM)
        FloatArray2D mass,             // Input mass, Shape (batch_size, N)
        FloatArray2D dL_rho_out,       // Gradient of output density, Shape (batch_size, N)
        FloatArray3D grad_x,           // Backward gradient w.r.t to position Shape (batch_size, N, DIM)
        IntArray1D bin_offsets,        // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info,     // Shape (max_blocks, 4)
        bool dLdx = true               // Whether to compute gradient w.r.t. position
    )
    {
        if (!dLdx)
            return; // No gradient to compute

        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(mass, {m_config.batch_size, m_config.particle_count}, "mass");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(dL_rho_out, {m_config.batch_size, m_config.particle_count}, "grad_output_rho");
        validate_array_shape(grad_x, {m_config.batch_size, m_config.particle_count, m_dimension}, "grad_x");
        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        const float *d_mass = mass.data();
        ParticlePosition<DIM> *d_grad_x = reinterpret_cast<ParticlePosition<DIM> *>(grad_x.data());

        if constexpr (ST == KernelStrategy::GridBased)
        {
            BlockInfo *d_block_info = reinterpret_cast<BlockInfo *>(block_info.data());
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 block_dim(128, 1, 1);
            dim3 grid_dim(m_max_blocks, 1, 1);
            const int STRIDE = 128;
            size_t smem_size = STRIDE * (1 + 1 + 2 + DIM) * sizeof(float); // local_mass and neighbor_mass
            backward_kernel<DIM, ComputationType::Density, STRIDE><<<grid_dim, block_dim, smem_size>>>(
                d_binned_positions, // Input binned positions
                nullptr,            // No input density
                d_mass,             // Input mass
                nullptr,            // No input state
                dL_rho_out.data(),  // Gradient of output density
                nullptr,            // No gradient of output count
                nullptr,            // No gradient of output blur state
                nullptr,            // No gradient of state gradient
                nullptr,            // No gradient of density gradient
                nullptr,            // No gradient of moment matrix
                d_grad_x,           // Gradient w.r.t. position
                nullptr,            // No backward w.r.t. rho
                nullptr,            // No backward w.r.t. state
                0,                  // No feature dim
                total_blocks,       // Total blocks used
                d_bin_offsets,      // Bin offsets
                d_block_info,       // Block info
                m_config);
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
            vanilla_backward_kernel<DIM, ComputationType::Density, 1><<<grid_dim, block_dim>>>(
                d_binned_positions, // Input binned positions
                nullptr,            // No input density
                d_mass,             // Input mass
                nullptr,            // No input state
                dL_rho_out.data(),  // Gradient of output density
                nullptr,            // No gradient of output count
                nullptr,            // No gradient of output blur state
                nullptr,            // No gradient of state gradient
                nullptr,            // No gradient of density gradient
                nullptr,            // No gradient of moment matrix
                d_grad_x,           // Gradient w.r.t. position
                nullptr,            // No backward w.r.t. rho
                nullptr,            // No backward w.r.t. state
                0,                  // No feature dim
                d_bin_offsets,      // Bin offsets
                m_config);
        }
    }

    void blur_backward(
        FloatArray3D binned_positions,  // Shape (batch_size, N, DIM)
        FloatArray2D rho,               // Input density, Shape (batch_size, N)
        FloatArray2D mass,              // Input mass, Shape (batch_size, N)
        FloatArray3D state_in,          // Input state, Shape (batch_size, N, feature_dim)
        FloatArray3D dL_blur_state_out, // Gradient of output state, Shape (batch_size, N, feature_dim)
        FloatArray3D grad_x,            // Backward gradient w.r.t to position Shape (batch_size, N, DIM)
        FloatArray3D grad_s,            // Backward gradient w.r.t to input state Shape (batch_size, N, feature_dim)
        FloatArray2D grad_rho,          // Backward gradient w.r.t to input density Shape (batch_size, N)
        IntArray1D bin_offsets,         // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info,      // Shape (max_blocks, 4)
        bool dLdx = true,               // Whether to compute gradient w.r.t. position
        bool dLds = true,               // Whether to compute gradient w.r.t. input state
        bool dLdrho = true              // Whether to compute gradient w.r.t. input density
    )
    {

        if (!dLdx && !dLds && !dLdrho)
            return; // No gradient to compute

        uint32_t feature_dim = state_in.shape(2);
        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(mass, {m_config.batch_size, m_config.particle_count}, "mass");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(rho, {m_config.batch_size, m_config.particle_count}, "rho");
        validate_array_shape(state_in, {m_config.batch_size, m_config.particle_count, feature_dim}, "state_in");
        validate_array_shape(dL_blur_state_out, {m_config.batch_size, m_config.particle_count, feature_dim}, "dL_blur_state_out");
        if (dLdx)
            validate_array_shape(grad_x, {m_config.batch_size, m_config.particle_count, m_dimension}, "grad_x");
        if (dLds)
            validate_array_shape(grad_s, {m_config.batch_size, m_config.particle_count, feature_dim}, "grad_s");
        if (dLdrho)
            validate_array_shape(grad_rho, {m_config.batch_size, m_config.particle_count}, "grad_rho");
        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        const float *d_mass = mass.data();
        ParticlePosition<DIM> *d_grad_x;
        d_grad_x = reinterpret_cast<ParticlePosition<DIM> *>(grad_x.data());
        float *d_grad_rho = reinterpret_cast<float *>(grad_rho.data());
        float *d_grad_s = reinterpret_cast<float *>(grad_s.data());
        bool DS_ONLY = dLds && !dLdx && !dLdrho;

        if constexpr (ST == KernelStrategy::GridBased)
        {
            BlockInfo *d_block_info = reinterpret_cast<BlockInfo *>(block_info.data());
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim(m_max_blocks, 1, 1);
            const int STRIDE = 64;
            // Lambda to launch kernel with appropriate template parameter
            auto launch_kernel = [&](auto ds_only_flag)
            {
                size_t smem_size;

                if constexpr (ds_only_flag)
                    smem_size = STRIDE * (feature_dim * 2 + 2) * sizeof(float);
                else
                    smem_size = STRIDE * (feature_dim * 5 + DIM + 5) * sizeof(float);

                backward_kernel<DIM, ComputationType::Blur, STRIDE, BC, ST, ds_only_flag><<<grid_dim, block_dim, smem_size>>>(
                    d_binned_positions,       // Input binned positions
                    rho.data(),               // Input density
                    d_mass,                   // Input mass
                    state_in.data(),          // Input state
                    nullptr,                  // Gradient of output density
                    nullptr,                  // No gradient of output count
                    dL_blur_state_out.data(), // Gradient of output blur state
                    nullptr,                  // No gradient of state gradient
                    nullptr,                  // No gradient of density gradient
                    nullptr,                  // No gradient of moment matrix
                    d_grad_x,                 // Gradient w.r.t. position
                    d_grad_rho,               // Gradient w.r.t. rho
                    d_grad_s,                 // Gradient w.r.t. state
                    feature_dim,              // Feature dim
                    total_blocks,             // Total blocks used
                    d_bin_offsets,            // Bin offsets
                    d_block_info,             // Block info
                    m_config);
            };

            // Launch with appropriate template parameter
            if (DS_ONLY)
                launch_kernel(std::true_type{});
            else
                launch_kernel(std::false_type{});
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);

            auto launch_blur_backward_kernel = [&](auto ds_only_flag, auto max_features)
            {
                vanilla_backward_kernel<DIM, ComputationType::Blur, max_features, BC, ST, ds_only_flag><<<grid_dim, block_dim>>>(
                    d_binned_positions,       // Input binned positions
                    rho.data(),               // Input density
                    d_mass,                   // Input mass
                    state_in.data(),          // Input state
                    nullptr,                  // Gradient of output density
                    nullptr,                  // No gradient of output count
                    dL_blur_state_out.data(), // Gradient of output blur state
                    nullptr,                  // No gradient of state gradient
                    nullptr,                  // No gradient of density gradient
                    nullptr,                  // No gradient of moment matrix
                    d_grad_x,                 // Gradient w.r.t. position
                    d_grad_rho,               // Gradient w.r.t. rho
                    d_grad_s,                 // Gradient w.r.t. state
                    feature_dim,              // Feature dim
                    d_bin_offsets,            // Bin offsets
                    m_config);
            };

            // Launch with appropriate template parameter
            if (DS_ONLY)
            {
                if (feature_dim <= 8)
                    launch_blur_backward_kernel(std::true_type{}, std::integral_constant<int, 8>{});
                else if (feature_dim <= 16)
                    launch_blur_backward_kernel(std::true_type{}, std::integral_constant<int, 16>{});
                else if (feature_dim <= 32)
                    launch_blur_backward_kernel(std::true_type{}, std::integral_constant<int, 32>{});
                else if (feature_dim <= 48)
                    launch_blur_backward_kernel(std::true_type{}, std::integral_constant<int, 48>{});
                else if (feature_dim <= 64)
                    launch_blur_backward_kernel(std::true_type{}, std::integral_constant<int, 64>{});
                else
                    throw std::runtime_error("feature_dim exceeds maximum supported value of 32.");
            }
            else
            {
                if (feature_dim <= 8)
                    launch_blur_backward_kernel(std::false_type{}, std::integral_constant<int, 8>{});
                else if (feature_dim <= 16)
                    launch_blur_backward_kernel(std::false_type{}, std::integral_constant<int, 16>{});
                else if (feature_dim <= 32)
                    launch_blur_backward_kernel(std::false_type{}, std::integral_constant<int, 32>{});
                else if (feature_dim <= 48)
                    launch_blur_backward_kernel(std::false_type{}, std::integral_constant<int, 48>{});
                else if (feature_dim <= 64)
                    launch_blur_backward_kernel(std::false_type{}, std::integral_constant<int, 64>{});
                else
                    throw std::runtime_error("feature_dim exceeds maximum supported value of 32.");
            }
        }
    }

    void density_gradient_backward(
        FloatArray3D binned_positions,        // Shape (batch_size, N, DIM)
        FloatArray2D mass,                    // Input mass, Shape (batch_size, N)
        FloatArray3D dL_density_gradient_out, // Gradient of output density gradient, Shape (batch_size, N, DIM)
        FloatArray3D grad_x,                  // Backward gradient w.r.t to position Shape (batch_size, N, DIM)
        IntArray1D bin_offsets,               // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info,            // Shape (max_blocks, 4)
        bool dLdx = true                      // Whether to compute gradient w.r.t. position
    )
    {
        if (!dLdx)
            return; // No gradient to compute

        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(mass, {m_config.batch_size, m_config.particle_count}, "mass");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(dL_density_gradient_out, {m_config.batch_size, m_config.particle_count, DIM}, "dL_density_gradient_out");
        validate_array_shape(grad_x, {m_config.batch_size, m_config.particle_count, m_dimension}, "grad_x");
        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        const float *d_mass = mass.data();
        ParticlePosition<DIM> *d_grad_x;
        d_grad_x = reinterpret_cast<ParticlePosition<DIM> *>(grad_x.data());

        if constexpr (ST == KernelStrategy::GridBased)
        {
            BlockInfo *d_block_info = reinterpret_cast<BlockInfo *>(block_info.data());
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 block_dim(128, 1, 1);
            dim3 grid_dim(m_max_blocks, 1, 1);
            const int STRIDE = 128;
            size_t smem_size = STRIDE * (DIM + DIM + DIM + 2) * sizeof(float); // local_mass and neighbor_mass
            backward_kernel<DIM, ComputationType::DensityGradient, STRIDE><<<grid_dim, block_dim, smem_size>>>(
                d_binned_positions,             // Input binned positions
                nullptr,                        // No input density
                d_mass,                         // Input mass
                nullptr,                        // No input state
                nullptr,                        // Gradient of output density
                nullptr,                        // No gradient of output count
                nullptr,                        // No gradient of output blur state
                nullptr,                        // No gradient of state gradient
                dL_density_gradient_out.data(), // No gradient of density gradient
                nullptr,                        // No gradient of moment matrix
                d_grad_x,                       // Gradient w.r.t. position
                nullptr,                        // No backward w.r.t. rho
                nullptr,                        // No backward w.r.t. state
                0,                              // No feature dim
                total_blocks,                   // Total blocks used
                d_bin_offsets,                  // Bin offsets
                d_block_info,                   // Block info
                m_config);
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
            vanilla_backward_kernel<DIM, ComputationType::DensityGradient, 1><<<grid_dim, block_dim>>>(
                d_binned_positions,             // Input binned positions
                nullptr,                        // No input density
                d_mass,                         // Input mass
                nullptr,                        // No input state
                nullptr,                        // Gradient of output density
                nullptr,                        // No gradient of output count
                nullptr,                        // No gradient of output blur state
                nullptr,                        // No gradient of state gradient
                dL_density_gradient_out.data(), // No gradient of density gradient
                nullptr,                        // No gradient of moment matrix
                d_grad_x,                       // Gradient w.r.t. position
                nullptr,                        // No backward w.r.t. rho
                nullptr,                        // No backward w.r.t. state
                0,                              // No feature dim
                d_bin_offsets,                  // Bin offsets
                m_config);
        }
    }

    void gradient_backward(
        FloatArray3D binned_positions,  // Shape (batch_size, N, DIM)
        FloatArray2D rho,               // Input density, Shape (batch_size, N)
        FloatArray2D mass,              // Input mass, Shape (batch_size, N)
        FloatArray3D state_in,          // Input state, Shape (batch_size, N, feature_dim)
        FloatArray4D dL_grad_state_out, // Gradient of output state, Shape (batch_size, N, feature_dim, DIM)
        FloatArray3D grad_x,            // Backward gradient w.r.t to position Shape (batch_size, N, DIM)
        FloatArray3D grad_s,            // Backward gradient w.r.t to input state Shape (batch_size, N, feature_dim)
        FloatArray2D grad_rho,          // Backward gradient w.r.t to input density Shape (batch_size, N)
        IntArray1D bin_offsets,         // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info,      // Shape (max_blocks, 4)
        bool dLdx = true,               // Whether to compute gradient w.r.t. position
        bool dLds = true,               // Whether to compute gradient w.r.t. input state
        bool dLdrho = true              // Whether to compute gradient w.r.t. input density
    )
    {
        if (!dLdx && !dLds && !dLdrho)
            return; // No gradient to compute

        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(mass, {m_config.batch_size, m_config.particle_count}, "mass");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(rho, {m_config.batch_size, m_config.particle_count}, "rho");
        uint32_t feature_dim = state_in.shape(2);
        validate_array_shape(state_in, {m_config.batch_size, m_config.particle_count, feature_dim}, "state_in");
        validate_array_shape(dL_grad_state_out, {m_config.batch_size, m_config.particle_count, feature_dim, DIM}, "dL_grad_state_out");
        if (dLdx)
            validate_array_shape(grad_x, {m_config.batch_size, m_config.particle_count, m_dimension}, "grad_x");
        if (dLds)
            validate_array_shape(grad_s, {m_config.batch_size, m_config.particle_count, feature_dim}, "grad_s");
        if (dLdrho)
            validate_array_shape(grad_rho, {m_config.batch_size, m_config.particle_count}, "grad_rho");

        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        const float *d_mass = mass.data();
        ParticlePosition<DIM> *d_grad_x;
        d_grad_x = reinterpret_cast<ParticlePosition<DIM> *>(grad_x.data());
        float *d_grad_rho = reinterpret_cast<float *>(grad_rho.data());
        float *d_grad_s = reinterpret_cast<float *>(grad_s.data());
        bool DS_ONLY = dLds && !dLdx && !dLdrho;

        if constexpr (ST == KernelStrategy::GridBased)
        {
            BlockInfo *d_block_info = reinterpret_cast<BlockInfo *>(block_info.data());
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim(m_max_blocks, 1, 1);
            const int STRIDE = 32;
            auto launch_kernel = [&](auto ds_only_flag)
            {
                size_t smem_size;
                if constexpr (ds_only_flag)
                    // Optimized memory layout for DS_ONLY case
                    smem_size = STRIDE * (feature_dim * (2 * DIM + 1) + 4) * sizeof(float);
                else
                    // Full memory layout for all gradients
                    smem_size = STRIDE * (feature_dim * (3 + 2 * DIM) + DIM + 5) * sizeof(float);

                backward_kernel<DIM, ComputationType::Gradient, STRIDE, BC, ST, ds_only_flag><<<grid_dim, block_dim, smem_size>>>(
                    d_binned_positions,       // Input binned positions
                    rho.data(),               // Input density
                    d_mass,                   // Input mass
                    state_in.data(),          // Input state
                    nullptr,                  // Gradient of output density
                    nullptr,                  // No gradient of output count
                    nullptr,                  // No gradient of output blur state
                    dL_grad_state_out.data(), // Gradient of state gradient
                    nullptr,                  // No gradient of density gradient
                    nullptr,                  // No gradient of moment matrix
                    d_grad_x,                 // Gradient w.r.t. position
                    d_grad_rho,               // Gradient w.r.t. rho
                    d_grad_s,                 // Gradient w.r.t. state
                    feature_dim,              // Feature dim
                    total_blocks,             // Total blocks used
                    d_bin_offsets,            // Bin offsets
                    d_block_info,             // Block info
                    m_config);
            };

            // Launch with appropriate template parameter
            if (DS_ONLY)
                launch_kernel(std::true_type{});
            else
                launch_kernel(std::false_type{});
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);

            auto launch_gradient_backward_kernel = [&](auto ds_only_flag, auto max_features)
            {
                vanilla_backward_kernel<DIM, ComputationType::Gradient, max_features, BC, ST, ds_only_flag><<<grid_dim, block_dim>>>(
                    d_binned_positions,       // Input binned positions
                    rho.data(),               // Input density
                    d_mass,                   // Input mass
                    state_in.data(),          // Input state
                    nullptr,                  // Gradient of output density
                    nullptr,                  // No gradient of output count
                    nullptr,                  // No gradient of output blur state
                    dL_grad_state_out.data(), // Gradient of state gradient
                    nullptr,                  // No gradient of density gradient
                    nullptr,                  // No gradient of moment matrix
                    d_grad_x,                 // Gradient w.r.t. position
                    d_grad_rho,               // Gradient w.r.t. rho
                    d_grad_s,                 // Gradient w.r.t. state
                    feature_dim,              // Feature dim
                    d_bin_offsets,            // Bin offsets
                    m_config);
            };

            // Launch with appropriate template parameter
            if (DS_ONLY)
            {
                if (feature_dim <= 8)
                    launch_gradient_backward_kernel(std::true_type{}, std::integral_constant<int, 8>{});
                else if (feature_dim <= 16)
                    launch_gradient_backward_kernel(std::true_type{}, std::integral_constant<int, 16>{});
                else if (feature_dim <= 32)
                    launch_gradient_backward_kernel(std::true_type{}, std::integral_constant<int, 32>{});
                else if (feature_dim <= 48)
                    launch_gradient_backward_kernel(std::true_type{}, std::integral_constant<int, 48>{});
                else if (feature_dim <= 64)
                    launch_gradient_backward_kernel(std::true_type{}, std::integral_constant<int, 64>{});
                else
                    throw std::runtime_error("feature_dim exceeds maximum supported value of 32.");
            }
            else
            {
                if (feature_dim <= 8)
                    launch_gradient_backward_kernel(std::false_type{}, std::integral_constant<int, 8>{});
                else if (feature_dim <= 16)
                    launch_gradient_backward_kernel(std::false_type{}, std::integral_constant<int, 16>{});
                else if (feature_dim <= 32)
                    launch_gradient_backward_kernel(std::false_type{}, std::integral_constant<int, 32>{});
                else if (feature_dim <= 48)
                    launch_gradient_backward_kernel(std::false_type{}, std::integral_constant<int, 48>{});
                else if (feature_dim <= 64)
                    launch_gradient_backward_kernel(std::false_type{}, std::integral_constant<int, 64>{});
                else
                    throw std::runtime_error("feature_dim exceeds maximum supported value of 32.");
            }
        }
    }

    void moment_matrix_backward(
        FloatArray3D binned_positions,     // Shape (batch_size, N, DIM)
        FloatArray2D rho,                  // Input density, Shape (batch_size, N)
        FloatArray2D mass,                 // Input mass, Shape (batch_size, N)
        FloatArray4D dL_moment_matrix_out, // Gradient of output moment matrix, Shape (batch_size, N, DIM, DIM)
        FloatArray3D grad_x,               // Backward gradient w.r.t to position Shape (batch_size, N, DIM)
        FloatArray2D grad_rho,             // Backward gradient w.r.t to input density Shape (batch_size, N)
        IntArray1D bin_offsets,            // Shape (batch_size * cell_count + 1)
        BlockInfoArray block_info,         // Shape (max_blocks, 4)
        bool dLdx = true,                  // Whether to compute gradient w.r.t. position
        bool dLdrho = true                 // Whether to compute gradient w.r.t. input density
    )
    {
        if (!dLdx && !dLdrho)
            return; // No gradient to compute

        validate_array_shape(binned_positions, {m_config.batch_size, m_config.particle_count, m_dimension}, "binned_positions");
        validate_array_shape(mass, {m_config.batch_size, m_config.particle_count}, "mass");
        validate_array_shape(bin_offsets, {m_config.batch_size * m_config.cell_count + 1}, "bin_offsets");
        validate_array_shape(rho, {m_config.batch_size, m_config.particle_count}, "rho");
        validate_array_shape(dL_moment_matrix_out, {m_config.batch_size, m_config.particle_count, DIM, DIM}, "dL_moment_matrix_out");
        if (dLdx)
            validate_array_shape(grad_x, {m_config.batch_size, m_config.particle_count, m_dimension}, "grad_x");
        if (dLdrho)
            validate_array_shape(grad_rho, {m_config.batch_size, m_config.particle_count}, "grad_rho");
        if constexpr (ST == KernelStrategy::GridBased)
            validate_array_shape(block_info, {m_max_blocks, 4}, "block_info");

        // Cast device pointers
        const ParticlePosition<DIM> *d_binned_positions = reinterpret_cast<const ParticlePosition<DIM> *>(binned_positions.data());
        const uint32_t *d_bin_offsets = reinterpret_cast<const uint32_t *>(bin_offsets.data());
        const float *d_mass = mass.data();
        ParticlePosition<DIM> *d_grad_x;
        d_grad_x = reinterpret_cast<ParticlePosition<DIM> *>(grad_x.data());
        float *d_grad_rho = reinterpret_cast<float *>(grad_rho.data());

        if constexpr (ST == KernelStrategy::GridBased)
        {
            BlockInfo *d_block_info = reinterpret_cast<BlockInfo *>(block_info.data());
            uint32_t *total_blocks = &m_block_offsets[m_config.batch_size * m_config.cell_count];
            dim3 block_dim(128, 1, 1);
            dim3 grid_dim(m_max_blocks, 1, 1);
            const int STRIDE = 128;
            size_t smem_size = STRIDE * (DIM * DIM * 2 + 2 + DIM + 3) * sizeof(float);
            backward_kernel<DIM, ComputationType::MomentMatrix, STRIDE><<<grid_dim, block_dim, smem_size>>>(
                d_binned_positions,          // Input binned positions
                rho.data(),                  // No input density
                d_mass,                      // Input mass
                nullptr,                     // No input state
                nullptr,                     // Gradient of output density
                nullptr,                     // No gradient of output count
                nullptr,                     // No gradient of output blur state
                nullptr,                     // No gradient of state gradient
                nullptr,                     // No gradient of density gradient
                dL_moment_matrix_out.data(), // No gradient of moment matrix
                d_grad_x,                    // Gradient w.r.t. position
                d_grad_rho,                  // Gradient w.r.t. rho
                nullptr,                     // No backward w.r.t. state
                0,                           // No feature dim
                total_blocks,                // Total blocks used
                d_bin_offsets,               // Bin offsets
                d_block_info,                // Block info
                m_config);
        }
        else
        {
            dim3 block_dim(256, 1, 1);
            dim3 grid_dim((m_config.particle_count + block_dim.x - 1) / block_dim.x, m_config.batch_size, 1);
            vanilla_backward_kernel<DIM, ComputationType::MomentMatrix, 1><<<grid_dim, block_dim>>>(
                d_binned_positions,          // Input binned positions
                rho.data(),                  // No input density
                d_mass,                      // Input mass
                nullptr,                     // No input state
                nullptr,                     // Gradient of output density
                nullptr,                     // No gradient of output count
                nullptr,                     // No gradient of output blur state
                nullptr,                     // No gradient of state gradient
                nullptr,                     // No gradient of density gradient
                dL_moment_matrix_out.data(), // No gradient of moment matrix
                d_grad_x,                    // Gradient w.r.t. position
                d_grad_rho,                  // Gradient w.r.t. rho
                nullptr,                     // No backward w.r.t. state
                0,                           // No feature dim
                d_bin_offsets,               // Bin offsets
                m_config);
        }
    }
};

template <BoundaryCondition BC, KernelStrategy ST>
class HashGrid2D : public HashGrid<2, BC, ST>
{
public:
    HashGrid2D(int grid_size_x, int grid_size_y, float eps, uint32_t particle_count, uint32_t batch_size, uint32_t max_particles_per_block)
    {
        this->m_config.grid_size[0] = grid_size_x;
        this->m_config.grid_size[1] = grid_size_y;
        this->m_config.cell_count = (uint32_t)(grid_size_x * grid_size_y);
        this->m_config.is_pow2 = (grid_size_x & (grid_size_x - 1)) == 0 && (grid_size_y & (grid_size_y - 1)) == 0;

        this->_init(eps, particle_count, batch_size, max_particles_per_block);
    }
};

using HashGrid2DPeriodic = HashGrid2D<BoundaryCondition::PERIODIC, KernelStrategy::GridBased>;
using HashGrid2DBox = HashGrid2D<BoundaryCondition::CLAMP, KernelStrategy::GridBased>;

using VHashGrid2DPeriodic = HashGrid2D<BoundaryCondition::PERIODIC, KernelStrategy::ParticleBased>;
using VHashGrid2DBox = HashGrid2D<BoundaryCondition::CLAMP, KernelStrategy::ParticleBased>;

template <BoundaryCondition BC, KernelStrategy ST>
class HashGrid3D : public HashGrid<3, BC, ST>
{
public:
    HashGrid3D(int grid_size_x, int grid_size_y, int grid_size_z, float eps, uint32_t particle_count, uint32_t batch_size, uint32_t max_particles_per_block)
    {
        this->m_config.grid_size[0] = grid_size_x;
        this->m_config.grid_size[1] = grid_size_y;
        this->m_config.grid_size[2] = grid_size_z;
        this->m_config.cell_count = (uint32_t)(grid_size_x * grid_size_y * grid_size_z);
        this->m_config.is_pow2 = (grid_size_x & (grid_size_x - 1)) == 0 && (grid_size_y & (grid_size_y - 1)) == 0 &&
                                 (grid_size_z & (grid_size_z - 1)) == 0;

        this->_init(eps, particle_count, batch_size, max_particles_per_block);
    }

    int grid_size_z() const { return this->m_config.grid_size[2]; }
};

using HashGrid3DPeriodic = HashGrid3D<BoundaryCondition::PERIODIC, KernelStrategy::GridBased>;
using HashGrid3DBox = HashGrid3D<BoundaryCondition::CLAMP, KernelStrategy::GridBased>;

using VHashGrid3DPeriodic = HashGrid3D<BoundaryCondition::PERIODIC, KernelStrategy::ParticleBased>;
using VHashGrid3DBox = HashGrid3D<BoundaryCondition::CLAMP, KernelStrategy::ParticleBased>;

void __global__ safe_inv_2x2_sym_mat_kernel(
    const float4 * __restrict__ in_matrix, // matrix of shape [B, N, 2, 2]
    float4 * __restrict__ out_matrix,      // matrix of shape [B, N, 2, 2]
    uint32_t batch_size,
    uint32_t N,
    uint32_t dim,
    float tol = 1e-3f)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch_size * N;
    if (idx >= total)
        return;

    float4 in_mat = in_matrix[idx];

    float det = in_mat.x * in_mat.w - in_mat.y * in_mat.y; // ad - b^2
    if (fabsf(det) < tol)
    {
        // Set to identity if near-singular
        out_matrix[idx] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    else
    {
        float inv_det = 1.0f / det;
        // Inverse of 2x2 matrix [a b; b d] is (1/det) * [d -b; -b a]
        out_matrix[idx] = make_float4(in_mat.w * inv_det, -in_mat.y * inv_det,
                                      -in_mat.y * inv_det, in_mat.x * inv_det);
    }
}

void __global__ safe_inv_3x3_sym_mat_kernel(
    const float * __restrict__ in_matrix, // matrix of shape [B, N, 3, 3]
    float * __restrict__ out_matrix,      // matrix of shape [B, N, 3, 3]
    uint32_t batch_size,
    uint32_t N,
    uint32_t dim,
    float tol = 1e-3f)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch_size * N;
    if (idx >= total)
        return;

    // a b c
    // b d e
    // c e f
    float a = in_matrix[idx * 9 + 0];
    float b = in_matrix[idx * 9 + 1];
    float c = in_matrix[idx * 9 + 2];
    float d = in_matrix[idx * 9 + 4];
    float e = in_matrix[idx * 9 + 5];
    float f = in_matrix[idx * 9 + 8];
    // Compute determinant
    float t1 = d * f - e * e; // uses symmetry
    float t2 = c * e - b * f;
    float t3 = b * e - c * d;
    float det = a * t1 + b * t2 + c * t3;
    if (fabsf(det) < tol)
    {
        // Set to identity if near-singular
        out_matrix[idx * 9 + 0] = 1.0f;
        out_matrix[idx * 9 + 1] = 0.0f;
        out_matrix[idx * 9 + 2] = 0.0f;
        out_matrix[idx * 9 + 3] = 0.0f;
        out_matrix[idx * 9 + 4] = 1.0f;
        out_matrix[idx * 9 + 5] = 0.0f;
        out_matrix[idx * 9 + 6] = 0.0f;
        out_matrix[idx * 9 + 7] = 0.0f;
        out_matrix[idx * 9 + 8] = 1.0f;
    }
    else
    {
        float inv_det = 1.0f / det;
        // Compute inverse using formula for symmetric matrix
        out_matrix[idx * 9 + 0] = t1 * inv_det;              // (d*f - e*e) / det
        out_matrix[idx * 9 + 1] = t2 * inv_det;              // (c*e - b*f) / det
        out_matrix[idx * 9 + 2] = t3 * inv_det;              // (b*e - c*d) / det
        out_matrix[idx * 9 + 3] = t2 * inv_det;              // symmetry
        out_matrix[idx * 9 + 4] = (a * f - c * c) * inv_det; // (a*f - c*c) / det
        out_matrix[idx * 9 + 5] = (b * c - a * e) * inv_det; // (b*c - a*e) / det
        out_matrix[idx * 9 + 6] = t3 * inv_det;              // symmetry
        out_matrix[idx * 9 + 7] = (b * c - a * e) * inv_det; // symmetry
        out_matrix[idx * 9 + 8] = (a * d - b * b) * inv_det; // (a*d - b*b) / det
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
    // dim should be either 2 or 3
    if (dim != 2 && dim != 3)
    {
        throw std::runtime_error("safe_inverse only supports 2x2 or 3x3 matrices.");
    }
    validate_array_shape(in_matrix, {batch_size, N, dim, dim}, "in_matrix");
    validate_array_shape(out_matrix, {batch_size, N, dim, dim}, "out_matrix");

    uint32_t total = batch_size * N;
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim((total + block_dim.x - 1) / block_dim.x, 1, 1);

    if (dim == 2)
    {
        const float4 *in_matrix_ptr = reinterpret_cast<const float4 *>(in_matrix.data());
        float4 *out_matrix_ptr = reinterpret_cast<float4 *>(out_matrix.data());

        safe_inv_2x2_sym_mat_kernel<<<grid_dim, block_dim>>>(
            in_matrix_ptr,
            out_matrix_ptr,
            batch_size,
            N,
            2,
            tol);
    }
    else
    {
        const float *in_matrix_ptr = reinterpret_cast<const float *>(in_matrix.data());
        float *out_matrix_ptr = reinterpret_cast<float *>(out_matrix.data());

        safe_inv_3x3_sym_mat_kernel<<<grid_dim, block_dim>>>(
            in_matrix_ptr,
            out_matrix_ptr,
            batch_size,
            N,
            3,
            tol);
    }
}

// Macro to add common properties and methods
#define ADD_COMMON_GRID_BINDINGS(grid_class, GridType, pos_shape)                                                                 \
    grid_class                                                                                                                    \
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
                                    "Initialize a 2D hash grid with given resolution, cell size, particle count, and batch size. Grid sizes must be powers of two.");
    ADD_COMMON_GRID_BINDINGS(grid2d_periodic, HashGrid2DPeriodic, "(batch_size, N, 2)");

    auto grid2d_box = nb::class_<HashGrid2DBox>(m, "HashGrid2DBox")
                          .def(nb::init<int, int, float, uint32_t, uint32_t, uint32_t>(),
                               "grid_size_x"_a, "grid_size_y"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                               "Initialize a 2D hash grid with given resolution, cell size, particle count, and batch size. Grid sizes must be powers of two.");
    ADD_COMMON_GRID_BINDINGS(grid2d_box, HashGrid2DBox, "(batch_size, N, 2)");

    // Vanilla 2D HashGrid
    auto vgrid2d_periodic = nb::class_<VHashGrid2DPeriodic>(m, "VHashGrid2DPeriodic")
                                .def(nb::init<int, int, float, uint32_t, uint32_t, uint32_t>(),
                                     "grid_size_x"_a, "grid_size_y"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                                     "Initialize a vanilla 2D hash grid with given resolution, cell size, particle count, and batch size. Grid sizes must be powers of two.");
    ADD_COMMON_GRID_BINDINGS(vgrid2d_periodic, VHashGrid2DPeriodic, "(batch_size, N, 2)");

    auto vgrid2d_box = nb::class_<VHashGrid2DBox>(m, "VHashGrid2DBox")
                           .def(nb::init<int, int, float, uint32_t, uint32_t, uint32_t>(),
                                "grid_size_x"_a, "grid_size_y"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                                "Initialize a vanilla 2D hash grid with given resolution, cell size, particle count, and batch size. Grid sizes must be powers of two.");
    ADD_COMMON_GRID_BINDINGS(vgrid2d_box, VHashGrid2DBox, "(batch_size, N, 2)");

    // 3D HashGrid
    auto grid3d_periodic = nb::class_<HashGrid3DPeriodic>(m, "HashGrid3DPeriodic")
                               .def(nb::init<int, int, int, float, uint32_t, uint32_t, uint32_t>(),
                                    "grid_size_x"_a, "grid_size_y"_a, "grid_size_z"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                                    "Initialize a 3D hash grid with given resolution, cell size, particle count, and batch size. Grid sizes must be powers of two.")
                               .def_prop_ro("grid_size_z", &HashGrid3DPeriodic::grid_size_z, "Grid size in z dimension");
    ADD_COMMON_GRID_BINDINGS(grid3d_periodic, HashGrid3DPeriodic, "(batch_size, N, 4)");

    auto grid3d_box = nb::class_<HashGrid3DBox>(m, "HashGrid3DBox")
                          .def(nb::init<int, int, int, float, uint32_t, uint32_t, uint32_t>(),
                               "grid_size_x"_a, "grid_size_y"_a, "grid_size_z"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                               "Initialize a 3D hash grid with given resolution, cell size, particle count, and batch size. Grid sizes must be powers of two.")
                          .def_prop_ro("grid_size_z", &HashGrid3DBox::grid_size_z, "Grid size in z dimension");
    ADD_COMMON_GRID_BINDINGS(grid3d_box, HashGrid3DBox, "(batch_size, N, 4)");

    // Vanilla 3D HashGrid
    auto vgrid3d_periodic = nb::class_<VHashGrid3DPeriodic>(m, "VHashGrid3DPeriodic")
                                .def(nb::init<int, int, int, float, uint32_t, uint32_t, uint32_t>(),
                                     "grid_size_x"_a, "grid_size_y"_a, "grid_size_z"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                                     "Initialize a vanilla 3D hash grid with given resolution, cell size, particle count, and batch size. Grid sizes must be powers of two.")
                                .def_prop_ro("grid_size_z", &VHashGrid3DPeriodic::grid_size_z, "Grid size in z dimension");
    ADD_COMMON_GRID_BINDINGS(vgrid3d_periodic, VHashGrid3DPeriodic, "(batch_size, N, 4)");

    auto vgrid3d_box = nb::class_<VHashGrid3DBox>(m, "VHashGrid3DBox")
                           .def(nb::init<int, int, int, float, uint32_t, uint32_t, uint32_t>(),
                                "grid_size_x"_a, "grid_size_y"_a, "grid_size_z"_a, "eps"_a, "particle_count"_a, "batch_size"_a, "max_particles_per_block"_a = 64,
                                "Initialize a vanilla 3D hash grid with given resolution, cell size, particle count, and batch size. Grid sizes must be powers of two.")
                           .def_prop_ro("grid_size_z", &VHashGrid3DBox::grid_size_z, "Grid size in z dimension");
    ADD_COMMON_GRID_BINDINGS(vgrid3d_box, VHashGrid3DBox, "(batch_size, N, 4)");

    m.def("safe_inv_sym_mat", &safe_inv_sym_mat, "in_matrix"_a, "out_matrix"_a, "tol"_a,
          "Compute the safe inverse of a batch of 2x2 or 3x3 symmetric matrices."
          "If the matrix is near-singular, the output will be the identity matrix.");
}
