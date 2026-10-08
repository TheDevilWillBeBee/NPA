#pragma once
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>


#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

template <int Dim>
struct ParticlePosition;

template <>
struct ParticlePosition<2>
{
    float2 pos; // Position (x, y)

    // Allow access via index [uint32_t dim_idx] for generic programming
    __device__ __host__ float operator[](uint32_t dim_idx) const
    {
        if (dim_idx == 0)
            return pos.x;
        else if (dim_idx == 1)
            return pos.y;
        else
        {
            // Out of bounds access, handle as needed (here we just return pos.x)
            return pos.x; // or throw an error if preferred
        }
    }
};

template <>
struct ParticlePosition<3>
{
    float3 pos; // Position (x, y, z)

    // Allow access via index [uint32_t dim_idx] for generic programming
    __device__ __host__ float operator[](uint32_t dim_idx) const
    {
        if (dim_idx == 0)
            return pos.x;
        else if (dim_idx == 1)
            return pos.y;
        else if (dim_idx == 2)
            return pos.z;
        else
        {
            // Out of bounds access, handle as needed (here we just return pos.x)
            return pos.x; // or throw an error if preferred
        }
    }
};

template <int DIM>
__device__ ParticlePosition<DIM> read_position(const ParticlePosition<DIM> *positions, uint32_t particle_count,
                                               uint32_t batch_idx, uint32_t particle_idx)
{
    return positions[batch_idx * particle_count + particle_idx];
}

__device__ float read_state(const float *state, uint32_t particle_count, uint32_t feature_dim, uint32_t batch_idx,
                            uint32_t particle_idx, uint32_t feature_idx)
{
    return state[(batch_idx * particle_count + particle_idx) * feature_dim + feature_idx];
}

__device__ inline float read_mass(const float *mass, uint32_t particle_count, uint32_t batch_idx,
                                  uint32_t particle_idx, float default_mass)
{
    if (mass == nullptr)
        return default_mass;
    return mass[batch_idx * particle_count + particle_idx];
}

__device__ __forceinline__ float pow2(float x) { return x * x; }
__device__ __forceinline__ float pow3(float x) { return x * x * x; }

template <int DIM>
__device__ inline float dot(const ParticlePosition<DIM> &a, const ParticlePosition<DIM> &b)
{
    if constexpr (DIM == 2)
    {
        return a.pos.x * b.pos.x + a.pos.y * b.pos.y;
    }
    else if constexpr (DIM == 3)
    {
        return a.pos.x * b.pos.x + a.pos.y * b.pos.y + a.pos.z * b.pos.z;
    }
    return 0.0f; // Should never reach here
}

template <int DIM>
__device__ inline float smoothing_kernel(const ParticlePosition<DIM> &r, float eps)
{
    float d2 = dot(r, r);
    float eps2 = eps * eps;

    return fmaxf(0.0f, pow3(eps2 - d2));
}

template <int DIM>
__device__ inline void backward_smoothing_kernel_inplace(const ParticlePosition<DIM> &r, float eps, float coef, float *output)
{
    float d2 = dot(r, r);
    if (d2 == 0.0f)
        return;
    float eps2 = eps * eps;
    if (d2 < eps2)
    {
        float mag = -coef * 6.0f * pow2(eps2 - d2);
        for (uint32_t dim = 0; dim < DIM; ++dim)
            output[dim] += mag * r[dim];
    }
    return;
}

template <int DIM>
__device__ inline void spiky_kernel_inplace(const ParticlePosition<DIM> &r, float eps, float coeff, float *output)
{
    float d2 = dot(r, r);
    float d = sqrtf(d2);
    if (d == 0.0f)
        return;

    if (d < eps)
    {
        float mag = coeff * 3.0f * pow2(eps - d) / d;
        for (uint32_t dim = 0; dim < DIM; ++dim)
            output[dim] += mag * r[dim];
    }

    return;
}

template <int DIM>
__device__ inline void backward_spiky_kernel_inplace(const ParticlePosition<DIM> &r, float eps, float *grad_in, float *grad_out)
{
    float d2 = dot(r, r);
    float eps2 = eps * eps;
    if (d2 == 0.0f || d2 >= eps2)
        return;

    float d = sqrtf(d2);
    float di = 1.0f / d;
    float mag1 = -6.0f * (eps - d);
    float mag2 = 3.0f * (eps2 - d2) * pow3(di);
    for (uint32_t i = 0; i < DIM; ++i)
        for (uint32_t j = 0; j < DIM; ++j)
        {
            float c = (i == j) ? (mag1 + mag2 * (d2 - r[i] * r[i])) : -mag2 * r[i] * r[j];
            grad_out[i] += grad_in[j] * c;
        }
}

template <int DIM>
__device__ inline void moment_matrix_inplace(const ParticlePosition<DIM> &r, float eps, float coef, float *output)
{
    float d2 = dot(r, r);
    float d = sqrtf(d2);
    if (d == 0.0f)
        return;

    if (d < eps)
    {
        float mag = coef * 3.0f * pow2(eps - d) / d;
        for (uint32_t i = 0; i < DIM; ++i)
            for (uint32_t j = 0; j < DIM; ++j)
                output[i * DIM + j] += mag * r[i] * r[j];
    }

    return;
}


enum class BoundaryCondition
{
    CLAMP = 0,
    PERIODIC = 1
};

// Positions are read and written in place as contiguous (B, N, DIM) float tensors.
static_assert(sizeof(ParticlePosition<2>) == 2 * sizeof(float), "2D positions must be two packed floats");
static_assert(sizeof(ParticlePosition<3>) == 3 * sizeof(float), "3D positions must be three packed floats");

// Add these member variables to the HashGrid class
struct BlockInfo
{
    uint32_t cell_idx;           // Which cell this block belongs to
    uint32_t offset_in_cell;     // Starting particle offset within the cell
    uint32_t particle_count;     // Number of particles this block handles (up to 256)
    uint32_t batch_idx;          // Which batch this block belongs to
    uint32_t candidates_per_row; // Average neighbor candidates per row span of the cell (fused lane sharing)
    uint32_t reserved[3];        // Keeps the record at two 16-byte words
};
static_assert(sizeof(BlockInfo) == 32, "BlockInfo is two 16-byte words");
constexpr int BLOCK_INFO_WORDS = sizeof(BlockInfo) / sizeof(uint32_t);

// In-place inverse of a symmetric DIM x DIM matrix (row-major, full storage).
// Near-singular matrices (|det| < tolerance) become the identity: isolated or
// degenerate neighborhoods get no moment correction instead of an unstable one.
// Shared by gradient_hybrid's safe_inv_sym_mat and the fused forward kernel.
template <int DIM>
__device__ __forceinline__ void safe_inverse_symmetric(float *matrix, float tolerance)
{
    if constexpr (DIM == 2)
    {
        float a = matrix[0], b = matrix[1], d = matrix[3];
        float det = a * d - b * b;
        if (fabsf(det) < tolerance)
        {
            matrix[0] = matrix[3] = 1.0f;
            matrix[1] = matrix[2] = 0.0f;
        }
        else
        {
            float inv_det = 1.0f / det;
            matrix[0] = d * inv_det;
            matrix[1] = matrix[2] = -b * inv_det;
            matrix[3] = a * inv_det;
        }
    }
    else
    {
        float a = matrix[0], b = matrix[1], c = matrix[2];
        float d = matrix[4], e = matrix[5], f = matrix[8];
        float t1 = d * f - e * e;
        float t2 = c * e - b * f;
        float t3 = b * e - c * d;
        float det = a * t1 + b * t2 + c * t3;
        if (fabsf(det) < tolerance)
        {
            #pragma unroll
            for (int k = 0; k < 9; ++k)
                matrix[k] = k % 4 == 0 ? 1.0f : 0.0f;
        }
        else
        {
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

template <int DIM, BoundaryCondition BC>
struct GridConfig
{
    int grid_size[DIM];      // Grid resolution in each dimension
    float eps;               // Grid cell spacing (also interaction cutoff distance)
    float inv_eps;           // Precomputed 1/eps for faster division
    uint32_t cell_count;     // Total number of grid cells
    uint32_t particle_count; // Total number of particles
    uint32_t batch_size;     // Batch size for processing multiple sets of particles
    float smoothing_coef;
    float spiky_coef;
    uint32_t max_blocks_per_batch;
    uint32_t MAX_PARTICLES_PER_BLOCK;
    bool is_pow2;
    float default_mass;
    float L[DIM];
    float half_L[DIM];
    float inv_L[DIM];

    __device__ inline float wrap_periodic(float d, uint32_t dim_idx) const
    {
        // Wrap displacement into [-L/2, L/2] where L = grid_size[dim] * eps:
        // subtract the nearest whole number of periods (one round-to-nearest,
        // one fma; a displacement already inside the range is returned exactly).
        float Ld = L[dim_idx];
        if (Ld <= 0.0f)
            return d;
        return fmaf(-rintf(d * inv_L[dim_idx]), Ld, d);
    }

    __device__ inline ParticlePosition<DIM> displacement(const ParticlePosition<DIM> &from,
                                                         const ParticlePosition<DIM> &to) const
    {
        ParticlePosition<DIM> r;
        if constexpr (DIM == 2)
        {
            float dx = to.pos.x - from.pos.x;
            float dy = to.pos.y - from.pos.y;
            if constexpr (BC == BoundaryCondition::PERIODIC)
            {
                dx = wrap_periodic(dx, 0);
                dy = wrap_periodic(dy, 1);
            }
            r.pos = make_float2(dx, dy);
        }
        else if constexpr (DIM == 3)
        {
            float dx = to.pos.x - from.pos.x;
            float dy = to.pos.y - from.pos.y;
            float dz = to.pos.z - from.pos.z;
            if constexpr (BC == BoundaryCondition::PERIODIC)
            {
                dx = wrap_periodic(dx, 0);
                dy = wrap_periodic(dy, 1);
                dz = wrap_periodic(dz, 2);
            }
            r.pos = make_float3(dx, dy, dz);
        }
        return r;
    }

    __device__ inline float l2_distance(const ParticlePosition<DIM> &a, const ParticlePosition<DIM> &b) const
    {
        ParticlePosition<DIM> r = displacement(a, b);
        return dot(r, r);
    }

    __device__ inline int positive_mod(int i, int n) const
    {
        return is_pow2 ? i & (n - 1) : (i % n + n) % n;
    }

    __device__ inline bool is_in_bound(int *cell_idx) const
    {
        if constexpr (BC == BoundaryCondition::PERIODIC)
            return true; // Always in bound for periodic BC
        else
        {
            bool in_bound = (cell_idx[0] >= 0 && cell_idx[0] < grid_size[0] &&
                             cell_idx[1] >= 0 && cell_idx[1] < grid_size[1]);

            if constexpr (DIM == 3)
                in_bound = in_bound && (cell_idx[2] >= 0 && cell_idx[2] < grid_size[2]);

            return in_bound;
        }
    }

    __device__ inline void apply_boundary(int *cell_idx) const
    {
        if constexpr (DIM == 2)
        {
            if constexpr (BC == BoundaryCondition::CLAMP)
            {
                cell_idx[0] = min(max(cell_idx[0], 0), grid_size[0] - 1);
                cell_idx[1] = min(max(cell_idx[1], 0), grid_size[1] - 1);
            }
            else
            {
                cell_idx[0] = positive_mod(cell_idx[0], grid_size[0]);
                cell_idx[1] = positive_mod(cell_idx[1], grid_size[1]);
            }
        }
        else if constexpr (DIM == 3)
        {
            if constexpr (BC == BoundaryCondition::CLAMP)
            {
                cell_idx[0] = min(max(cell_idx[0], 0), grid_size[0] - 1);
                cell_idx[1] = min(max(cell_idx[1], 0), grid_size[1] - 1);
                cell_idx[2] = min(max(cell_idx[2], 0), grid_size[2] - 1);
            }
            else
            {
                cell_idx[0] = positive_mod(cell_idx[0], grid_size[0]);
                cell_idx[1] = positive_mod(cell_idx[1], grid_size[1]);
                cell_idx[2] = positive_mod(cell_idx[2], grid_size[2]);
            }
        }
    }

    __device__ inline void pos2cell(const ParticlePosition<DIM> &particle, int *cell_idx) const
    {
        if constexpr (DIM == 2)
        {
            cell_idx[0] = __float2int_rd(particle.pos.x * inv_eps) + grid_size[0] / 2;
            cell_idx[1] = __float2int_rd(particle.pos.y * inv_eps) + grid_size[1] / 2;
        }
        else if constexpr (DIM == 3)
        {
            cell_idx[0] = __float2int_rd(particle.pos.x * inv_eps) + grid_size[0] / 2;
            cell_idx[1] = __float2int_rd(particle.pos.y * inv_eps) + grid_size[1] / 2;
            cell_idx[2] = __float2int_rd(particle.pos.z * inv_eps) + grid_size[2] / 2;
        }
        apply_boundary(cell_idx);
    }

    // Cell index (within a batch) of a position.
    __device__ inline uint32_t cell_hash(const ParticlePosition<DIM> &particle) const
    {
        int cell[DIM];
        pos2cell(particle, cell);
        return cell2hash(cell);
    }

    __device__ inline uint32_t cell2hash(const int *cell_idx) const
    {
        if constexpr (DIM == 2)
            return static_cast<uint32_t>(cell_idx[1] * grid_size[0] + cell_idx[0]);
        else
            return static_cast<uint32_t>((cell_idx[2] * grid_size[1] + cell_idx[1]) * grid_size[0] + cell_idx[0]);
    }

    __device__ inline void hash2cell(uint32_t hash, int *cell_idx) const
    {
        if constexpr (DIM == 2)
        {
            cell_idx[0] = hash % grid_size[0];
            cell_idx[1] = hash / grid_size[0];
        }
        else
        {
            cell_idx[0] = hash % grid_size[0];
            cell_idx[1] = (hash / grid_size[0]) % grid_size[1];
            cell_idx[2] = hash / (grid_size[0] * grid_size[1]);
        }
    }

    // Neighbor cells of a cell, visited as NEIGHBOR_ROWS rows of three
    // x-adjacent cells. Cells are x-fastest in the bin offsets, so each row is
    // one contiguous span of binned particles; at a periodic x seam the wrapped
    // cell is a second, disjoint chunk (wrap_chunk = 1). Spans contain
    // candidates outside the kernel support.
    static constexpr int NEIGHBOR_ROWS = DIM == 2 ? 3 : 9;
    static constexpr int WRAP_CHUNKS = BC == BoundaryCondition::PERIODIC ? 2 : 1;

    // The y (and z) coordinates of neighbor row `row` of `cell`, with the
    // boundary applied; false if the row lies outside a clamped grid.
    __device__ __forceinline__ bool neighbor_row(const int *cell, int row, int *neighbor) const
    {
        neighbor[0] = cell[0];
        if constexpr (DIM == 2)
            neighbor[1] = cell[1] + row - 1;
        else
        {
            neighbor[1] = cell[1] + row / 3 - 1;
            neighbor[2] = cell[2] + row % 3 - 1;
        }
        if (!is_in_bound(neighbor))
            return false;
        apply_boundary(neighbor);
        return true;
    }

    // x range [first_x, last_x] of one chunk of a neighbor row: the (up to)
    // three adjacent cells, or the wrapped cell at a periodic seam; false if
    // the chunk does not exist.
    __device__ __forceinline__ bool row_x_range(const int *cell, int wrap_chunk, int &first_x, int &last_x) const
    {
        if (wrap_chunk != 0)
        {
            if (cell[0] == 0)
                first_x = last_x = grid_size[0] - 1;
            else if (cell[0] == grid_size[0] - 1)
                first_x = last_x = 0;
            else
                return false;
        }
        else
        {
            first_x = max(0, cell[0] - 1);
            last_x = min(grid_size[0] - 1, cell[0] + 1);
        }
        return true;
    }

    // Binned particles [start, end) of one chunk of a neighbor row, indexed over
    // all batches; false for an absent or empty chunk.
    __device__ __forceinline__ bool neighbor_span(const int *cell, unsigned int batch, int row, int wrap_chunk,
                                                  const uint32_t *bin_offsets, uint32_t &start, uint32_t &end) const
    {
        int neighbor[DIM], first_x, last_x;
        if (!neighbor_row(cell, row, neighbor) || !row_x_range(cell, wrap_chunk, first_x, last_x))
            return false;
        neighbor[0] = first_x;
        unsigned int first = cell2hash(neighbor);
        neighbor[0] = last_x;
        unsigned int last = cell2hash(neighbor);
        start = bin_offsets[batch * cell_count + first];
        end = bin_offsets[batch * cell_count + last + 1];
        return start != end;
    }
};

template <typename ArrayType>
void validate_array_shape(const ArrayType &array,
                          const std::vector<size_t> &expected_shape,
                          const std::string &array_name)
{
    // Check number of dimensions
    if (array.ndim() != expected_shape.size())
    {
        throw std::runtime_error(
            array_name + " must have " + std::to_string(expected_shape.size()) +
            " dimensions, got " + std::to_string(array.ndim()));
    }

    // Check each dimension
    for (size_t i = 0; i < expected_shape.size(); ++i)
    {
        if (array.shape(i) != expected_shape[i])
        {
            std::string expected_shape_str = "(";
            for (size_t j = 0; j < expected_shape.size(); ++j)
            {
                if (j > 0)
                    expected_shape_str += ", ";
                expected_shape_str += std::to_string(expected_shape[j]);
            }
            expected_shape_str += ")";

            std::string actual_shape_str = "(";
            for (size_t j = 0; j < array.ndim(); ++j)
            {
                if (j > 0)
                    actual_shape_str += ", ";
                actual_shape_str += std::to_string(array.shape(j));
            }
            actual_shape_str += ")";

            throw std::runtime_error(
                array_name + " must have shape " + expected_shape_str +
                ", got " + actual_shape_str);
        }
    }
}