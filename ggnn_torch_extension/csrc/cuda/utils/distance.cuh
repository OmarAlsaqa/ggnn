#pragma once // Use pragma once for modern C++ headers

// Correct the include path to point to our new host-side API header
#include "../../cpu/ggnn_cpu.h"
#include "../../ggnn_config.h"

#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <cmath> // For fabs and sqrtf

// Move the utility into the ggnn::cuda namespace for consistency
namespace ggnn {
namespace cuda {

/**
 * Distance calculates the distance/difference between a query vector and a base vector.
 * This is a device-side utility meant to be used within a CUDA kernel.
 */
template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_DIM_X,
          uint32_t DIST_ITEMS_PER_THREAD>
struct Distance {
  const uint32_t D;
  const DistanceMeasure measure;

  // only valid in thread 0, only needed if measure == Cosine
  ValueT r_query_norm;

  using AddrT = size_t;

  struct DistanceAndNorm {
    ValueT dist{0.0f};
    ValueT norm{0.0f};

    struct Sum {
      __host__ __device__ __forceinline__ DistanceAndNorm operator()(const DistanceAndNorm& a,
                                                                     const DistanceAndNorm& b) const
      {
        return {a.dist + b.dist, a.norm + b.norm};
      }
    };
  };

  using BlockReduceDist = cub::BlockReduce<ValueT, BLOCK_DIM_X>;
  using BlockReduceDistAndNorm = cub::BlockReduce<DistanceAndNorm, BLOCK_DIM_X>;

  union TempStorage {
    typename BlockReduceDist::TempStorage dist_temp_storage;
    typename BlockReduceDistAndNorm::TempStorage dist_and_norm_temp_storage;
  };

  const BaseT* d_base;
  BaseT r_query[DIST_ITEMS_PER_THREAD];

  TempStorage& s_temp_storage;
  __device__ __forceinline__ TempStorage& PrivateTmpStorage()
  {
    __shared__ TempStorage s_tmp;
    return s_tmp;
  }
  ValueT& s_dist;
  __device__ __forceinline__ ValueT& DistTmpStorage()
  {
    __shared__ ValueT s_dist;
    return s_dist;
  }

  __device__ __forceinline__ Distance(const uint32_t D, DistanceMeasure measure,
                                      const BaseT* d_base, const BaseT* d_query, const KeyT n)
      : D(D),
        measure(measure),
        d_base(d_base),
        s_temp_storage(PrivateTmpStorage()),
        s_dist(DistTmpStorage())
  {
    loadQueryPos(d_query + static_cast<AddrT>(n) * D);
  }

  __device__ __forceinline__ Distance(const uint32_t D, DistanceMeasure measure,
                                      const BaseT* d_base, const KeyT n)
      : D(D),
        measure(measure),
        d_base(d_base),
        s_temp_storage(PrivateTmpStorage()),
        s_dist(DistTmpStorage())
  {
    loadQueryPos(d_base + static_cast<AddrT>(n) * D);
  }

  __device__ __forceinline__ void loadQueryPos(const BaseT* d_query)
  {
    ValueT query_norm = 0.0f;
    for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
      r_query[item] = (read_dim < D) ? d_query[read_dim] : 0;
      if (measure == DistanceMeasure::Cosine)
        query_norm += static_cast<ValueT>(r_query[item]) * static_cast<ValueT>(r_query[item]);
    }
    if (measure == DistanceMeasure::Cosine) {
      r_query_norm = BlockReduceDist(s_temp_storage.dist_temp_storage).Sum(query_norm);
    }
  }

  __device__ __forceinline__ ValueT distance_synced(const KeyT other_id)
  {
    BaseT r_other[DIST_ITEMS_PER_THREAD];

    for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
      r_other[item] = (read_dim < D) ? d_base[static_cast<AddrT>(other_id) * D + read_dim] : 0;
    }
    ValueT dist{0.0f};
    if (measure == DistanceMeasure::Euclidean) {
      for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
        const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
        const ValueT diff =
            (read_dim < D) ? static_cast<ValueT>(r_other[item]) - static_cast<ValueT>(r_query[item])
                           : 0;
        dist += diff * diff;
      }
      dist = BlockReduceDist(s_temp_storage.dist_temp_storage).Sum(dist);
      if (!threadIdx.x)
        s_dist = dist;
    }
    if (measure == DistanceMeasure::Cosine) {
      DistanceAndNorm dist_and_norm{0.0f, 0.0f};
      for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
        const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
        dist_and_norm.dist +=
            (read_dim < D) ? static_cast<ValueT>(r_other[item]) * static_cast<ValueT>(r_query[item])
                           : 0;
        dist_and_norm.norm +=
            (read_dim < D) ? static_cast<ValueT>(r_other[item]) * static_cast<ValueT>(r_other[item])
                           : 0;
      }
      dist_and_norm = BlockReduceDistAndNorm(s_temp_storage.dist_and_norm_temp_storage)
                          .Reduce(dist_and_norm, DistanceAndNorm::Sum());
      if (!threadIdx.x) {
        const ValueT norm_sqr = r_query_norm * dist_and_norm.norm;
        s_dist = (norm_sqr > 0.0f) ? fabsf(1.0f - dist_and_norm.dist / sqrtf(norm_sqr)) : 1.0f;
      }
    }
    __syncthreads();

    return s_dist;
  }
};

} // namespace cuda
} // namespace ggnn