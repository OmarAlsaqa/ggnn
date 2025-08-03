#pragma once  // Use pragma once for modern C++ headers

#include "../utils/distance.cuh"
#include "../utils/k_best_list.cuh"
#include "../../cpu/ggnn_cpu.h"  // For GraphConfig and DistanceMeasure
#include "../../ggnn_config.h"

namespace ggnn::cuda::detail{

/**
 * Kernel struct for brute-force query.
 * This is a simple parameter pack with its device-side logic included.
 * It has NO host-side methods.
 */
template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE, bool WRITE_DISTS>
struct BruteForceQueryKernel {
  static constexpr uint32_t BLOCK_DIM_X = BLOCK_SIZE;
  static constexpr uint32_t DIST_ITEMS_PER_THREAD = 4;

  // --- Member Variables (Kernel Parameters) ---
  uint32_t D;
  DistanceMeasure measure;
  uint32_t KQuery;
  KeyT N_base;
  const BaseT* d_base;
  const BaseT* d_query;
  KeyT* d_query_results;
  ValueT* d_query_results_dists;

  // --- CONSTRUCTOR ---
  BruteForceQueryKernel(uint32_t D, DistanceMeasure measure, uint32_t KQuery, KeyT N_base,
                        const BaseT* d_base, const BaseT* d_query, KeyT* d_query_results,
                        ValueT* d_query_results_dists)
      : D(D),
        measure(measure),
        KQuery(KQuery),
        N_base(N_base),
        d_base(d_base),
        d_query(d_query),
        d_query_results(d_query_results),
        d_query_results_dists(d_query_results_dists)
  {
  }

  // --- Device-Side Implementation ---
  // The logic from the original bf_query_layer.cu is now moved directly inside the struct.
  __device__ void operator()()
  {
    using Distance =
        ::ggnn::cuda::Distance<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD>;
    using KBestList = ::ggnn::cuda::KBestList<KeyT, ValueT, BLOCK_DIM_X>;

    const KeyT n = static_cast<KeyT>(blockIdx.x);

    // Distance calculator is thread-block specific
    Distance distCalc(D, measure, d_base, d_query, n);

    // K-Best list uses shared memory
    KBestList best(KQuery);
    __syncthreads();

    for (KeyT i = 0; i < N_base; ++i) {
      // Each thread block computes the distance for its assigned query to all base points.
      ValueT dist = distCalc.distance_synced(i);
      if (dist < best.worst()) {
        best.add_unique(dist, i);
      }
    }
    __syncthreads();

    // Write the final sorted list back to global memory
    for (uint32_t k = threadIdx.x; k < KQuery; k += BLOCK_DIM_X) {
      d_query_results[static_cast<size_t>(n) * KQuery + k] = best.s_ids[k];
      if constexpr (WRITE_DISTS) {
        d_query_results_dists[static_cast<size_t>(n) * KQuery + k] = best.s_dists[k];
      }
    }
  }
};

/**
 * This is the __global__ function that will be launched from our C++ host code.
 * It simply instantiates and calls the kernel struct's operator().
 */
template <typename KernelT>
__global__ void bf_query_kernel_launcher(KernelT kernel)
{
  kernel();
}

}  // namespace detail, cuda, ggnn