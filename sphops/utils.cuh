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

    __device__ __host__ ParticlePosition<2> operator+(const ParticlePosition<2> &other) const
    {
        // return __fadd2_rn(pos, other.pos);
        return {make_float2(pos.x + other.pos.x, pos.y + other.pos.y)};
    }

    __device__ __host__ ParticlePosition<2> operator-(const ParticlePosition<2> &other) const
    {
        // return {__fadd2_rn(pos, make_float2(-other.pos.x, -other.pos.y))};
        return {make_float2(pos.x - other.pos.x, pos.y - other.pos.y)};
    }

    __device__ __host__ ParticlePosition<2> &operator+=(const ParticlePosition<2> &other)
    {
        // this.pos = __fadd2_rn(this.pos, other.pos);
        pos.x += other.pos.x;
        pos.y += other.pos.y;
        return *this;
    }

    __device__ __host__ ParticlePosition<2> &operator-=(const ParticlePosition<2> &other)
    {
        // this.pos = __fadd2_rn(this.pos, make_float2(-other.pos.x, -other.pos.y));
        pos.x -= other.pos.x;
        pos.y -= other.pos.y;
        return *this;
    }

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
    float4 pos; // Position (x, y, z, w) - w can be unused or store extra data

    __device__ __host__ ParticlePosition<3> operator+(const ParticlePosition<3> &other) const
    {
        return {make_float4(pos.x + other.pos.x, pos.y + other.pos.y, pos.z + other.pos.z, 0.0f)};
    }

    __device__ __host__ ParticlePosition<3> operator-(const ParticlePosition<3> &other) const
    {
        return {make_float4(pos.x - other.pos.x, pos.y - other.pos.y, pos.z - other.pos.z, 0.0f)};
    }

    __device__ __host__ ParticlePosition<3> &operator+=(const ParticlePosition<3> &other)
    {
        pos.x += other.pos.x;
        pos.y += other.pos.y;
        pos.z += other.pos.z;
        return *this;
    }

    __device__ __host__ ParticlePosition<3> &operator-=(const ParticlePosition<3> &other)
    {
        pos.x -= other.pos.x;
        pos.y -= other.pos.y;
        pos.z -= other.pos.z;
        return *this;
    }

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
__device__ __forceinline__ float pow4(float x)
{
    float x2 = x * x;
    return x2 * x2;
}
__device__ __forceinline__ float pow5(float x)
{
    float x2 = x * x;
    return x2 * x2 * x;
}
__device__ __forceinline__ float pow6(float x)
{
    float x2 = x * x;
    float x3 = x2 * x;
    return x3 * x3;
}
__device__ __forceinline__ float pow8(float x)
{
    float x2 = x * x;
    float x4 = x2 * x2;
    return x4 * x4;
}

__device__ __forceinline__ float pow9(float x)
{
    float x2 = x * x;
    float x4 = x2 * x2;
    return x4 * x4 * x;
}

template <int DIM>
__device__ inline float l2_distance(const ParticlePosition<DIM> &a, const ParticlePosition<DIM> &b)
{
    if constexpr (DIM == 2)
    {
        float dx = a.pos.x - b.pos.x;
        float dy = a.pos.y - b.pos.y;
        return dx * dx + dy * dy;
    }
    else if constexpr (DIM == 3)
    {
        float dx = a.pos.x - b.pos.x;
        float dy = a.pos.y - b.pos.y;
        float dz = a.pos.z - b.pos.z;
        return dx * dx + dy * dy + dz * dz;
    }
    return 0.0f; // Should never reach here
}

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

enum class KernelStrategy
{
    GridBased,    // Advanced: Block-per-cell with shared memory
    ParticleBased // Vanilla: Thread-per-particle with registers
};

// Add these member variables to the HashGrid class
struct BlockInfo
{
    uint32_t cell_idx;       // Which cell this block belongs to
    uint32_t offset_in_cell; // Starting particle offset within the cell
    uint32_t particle_count; // Number of particles this block handles (up to 256)
    uint32_t batch_idx;      // Which batch this block belongs to
};

template <int DIM, BoundaryCondition BC, KernelStrategy ST>
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

    __device__ inline float wrap_periodic(float d, uint32_t dim_idx) const
    {
        // Wrap displacement into [-L/2, L/2) where L = grid_size[dim] * eps
        float Ld = L[dim_idx];
        if (Ld <= 0.0f)
            return d;
        float t = fmodf(d + half_L[dim_idx], Ld);
        if (t < 0.0f)
            t += Ld;
        return t - half_L[dim_idx];
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
            r.pos = make_float4(dx, dy, dz, 0.0f);
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
        if constexpr (ST == KernelStrategy::ParticleBased)
            return i & (n - 1); // n must be power of two
        else
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

    __device__ inline uint32_t cell2hash(const int *cell_idx) const
    {
        if constexpr (DIM == 2 && ST == KernelStrategy::GridBased)
            return static_cast<uint32_t>(cell_idx[1] * grid_size[0] + cell_idx[0]);
        else if constexpr (DIM == 3 && ST == KernelStrategy::GridBased)
            return static_cast<uint32_t>((cell_idx[2] * grid_size[1] + cell_idx[1]) * grid_size[0] + cell_idx[0]);
        else if constexpr (DIM == 2 && ST == KernelStrategy::ParticleBased)
            return static_cast<uint32_t>(dilate(cell_idx[0]) | (dilate(cell_idx[1]) << 1));
        else if constexpr (DIM == 3 && ST == KernelStrategy::ParticleBased)
            return static_cast<uint32_t>(dilate(cell_idx[0]) | (dilate(cell_idx[1]) << 1) | (dilate(cell_idx[2]) << 2));
        return 0; // Should never reach here
    }

    __device__ inline void hash2cell(uint32_t hash, int *cell_idx) const
    {
        if constexpr (DIM == 2 && ST == KernelStrategy::GridBased)
        {
            cell_idx[0] = hash % grid_size[0];
            cell_idx[1] = hash / grid_size[0];
        }
        else if constexpr (DIM == 3 && ST == KernelStrategy::GridBased)
        {
            cell_idx[0] = hash % grid_size[0];
            cell_idx[1] = (hash / grid_size[0]) % grid_size[1];
            cell_idx[2] = hash / (grid_size[0] * grid_size[1]);
        }
        else if constexpr (DIM == 2 && ST == KernelStrategy::ParticleBased)
        {
            cell_idx[0] = undilate(hash) & (grid_size[0] - 1);      // grid_size should be power of two
            cell_idx[1] = undilate(hash >> 1) & (grid_size[1] - 1); // grid_size should be power of two
        }
        else if constexpr (DIM == 3 && ST == KernelStrategy::ParticleBased)
        {
            cell_idx[0] = undilate(hash) & (grid_size[0] - 1);      // grid_size should be power of two
            cell_idx[1] = undilate(hash >> 1) & (grid_size[1] - 1); // grid_size should be power of two
            cell_idx[2] = undilate(hash >> 2) & (grid_size[2] - 1); // grid_size should be power of two
        }
    }

    __device__ inline int dilate(int x) const
    {
        if constexpr (DIM == 2)
        {
            x = x & 0x3FF;
            x = (x | (x << 16)) & 0x0000FFFF; // 16 bit gap
            x = (x | (x << 8)) & 0x00FF00FF;  // 8 bit gap
            x = (x | (x << 4)) & 0x0F0F0F0F;  // 4 bit gap
            x = (x | (x << 2)) & 0x33333333;  // 2 bit gap
            x = (x | (x << 1)) & 0x55555555;  // 1 bit gap
        }
        else if constexpr (DIM == 3)
        {
            x = x & 0x3FF;
            x = (x | (x << 16)) & 0x030000FF; // 16 bit gap (keep low 8 bits + 2 high bits)
            x = (x | (x << 8)) & 0x0300F00F;  // 8 bit gap
            x = (x | (x << 4)) & 0x030C30C3;  // 4 bit gap
            x = (x | (x << 2)) & 0x09249249;  // 2 bit gap (final pattern)
        }
        return x;
    }

    __device__ inline int undilate(int x) const
    {
        if constexpr (DIM == 2)
        {
            x = x & 0x55555555;
            x = (x | (x >> 1)) & 0x33333333;
            x = (x | (x >> 2)) & 0x0F0F0F0F;
            x = (x | (x >> 4)) & 0x00FF00FF;
            x = (x | (x >> 8)) & 0x0000FFFF;
            x = (x | (x >> 16)) & 0x000003FF; // keep only 10 bits (max index = 1023)
        }
        else if constexpr (DIM == 3)
        {
            x = x & 0x09249249;
            x = (x | (x >> 2)) & 0x030C30C3;
            x = (x | (x >> 4)) & 0x0300F00F;
            x = (x | (x >> 8)) & 0x030000FF;
            x = (x | (x >> 16)) & 0x000003FF; // keep only 10 bits
        }
        return x;
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