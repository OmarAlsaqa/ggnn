#pragma once

#include "../utils/simple_knn_cache.cuh"
#include "ggnn_cpu.h"

namespace ggnn {
namespace cuda {
namespace detail {

/**
 * Kernel struct for the main graph query operation.
 */
template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE, bool WRITE_DISTS,
          bool DIST_STATS>
struct QueryKernel {
  // --- Kernel Configuration ---
  static constexpr uint32_t BLOCK_DIM_X = BLOCK_SIZE;
  static constexpr uint32_t DIST_ITEMS_PER_THREAD = 4;

  // --- Member Variables (Kernel Parameters) ---
  uint32_t D;
  DistanceMeasure measure;
  uint32_t KQuery;
  uint32_t sorted_size;
  uint32_t cache_size;
  float tau_query;
  uint32_t max_iterations;
  const KeyT N_base;
  uint32_t KBuild;
  uint32_t num_starting_points;
  const BaseT* d_base;
  const BaseT* d_query;
  const KeyT* d_graph;
  const KeyT* d_starting_points;
  const float* d_nn1_stats;
  KeyT* d_query_results;
  ValueT* d_query_results_dists;
  const uint32_t* d_dist_stats;
  uint32_t shards_per_gpu;
  uint32_t on_gpu_shard_id;

  // --- CONSTRUCTOR ---
  QueryKernel(uint32_t D, DistanceMeasure measure, uint32_t KQuery, uint32_t sorted_size,
              uint32_t cache_size, float tau_query, uint32_t max_iterations, uint32_t N_base,
              uint32_t KBuild, uint32_t num_starting_points, const BaseT* d_base,
              const BaseT* d_query, const KeyT* d_graph, const KeyT* d_starting_points,
              const float* d_nn1_stats, KeyT* d_query_results, ValueT* d_query_results_dists,
              const uint32_t* d_dist_stats, uint32_t shards_per_gpu, uint32_t on_gpu_shard_id)
      : D(D),
        measure(measure),
        KQuery(KQuery),
        sorted_size(sorted_size),
        cache_size(cache_size),
        tau_query(tau_query),
        max_iterations(max_iterations),
        N_base(N_base),
        KBuild(KBuild),
        num_starting_points(num_starting_points),
        d_base(d_base),
        d_query(d_query),
        d_graph(d_graph),
        d_starting_points(d_starting_points),
        d_nn1_stats(d_nn1_stats),
        d_query_results(d_query_results),
        d_query_results_dists(d_query_results_dists),
        d_dist_stats(d_dist_stats),
        shards_per_gpu(shards_per_gpu),
        on_gpu_shard_id(on_gpu_shard_id)
  {
  }

  // --- Device-Side Implementation ---
  // (Implementation from original query_layer.cu)
  __device__ void operator()() /* remove const here */ {
    static constexpr uint32_t K_BLOCK = 32;

    using Cache = ::ggnn::cuda::SimpleKNNCache<KeyT, ValueT, BaseT, BLOCK_DIM_X,
                                               DIST_ITEMS_PER_THREAD, DIST_STATS>;

    const float xi = (measure == DistanceMeasure::Euclidean)
                         ? (d_nn1_stats[1] * d_nn1_stats[1]) * tau_query * tau_query
                         : d_nn1_stats[1] * tau_query;

    const KeyT n = static_cast<KeyT>(blockIdx.x);

    Cache cache(D, measure, KQuery, sorted_size, cache_size, d_base, d_query, n, xi);
    cache.fetch_unfiltered(d_starting_points, nullptr, num_starting_points);

    for (uint32_t ite = 0; ite < max_iterations; ++ite) {
      if (measure == DistanceMeasure::Euclidean) {
        cache.r_xi = min(xi, cache.s_dists[0] * tau_query * tau_query);
      }
      else if (measure == DistanceMeasure::Cosine) {
        cache.r_xi = min(xi, cache.s_dists[0] * tau_query);
      }

      const KeyT anchor = cache.pop();
      if (anchor == Cache::EMPTY_KEY) {
        break;
      }

      __shared__ KeyT s_knn[K_BLOCK];
      for (uint32_t i = 0; i < KBuild; i += K_BLOCK) {
        if (threadIdx.x < K_BLOCK) {
          s_knn[threadIdx.x] = (i + threadIdx.x < KBuild)
                                   ? d_graph[static_cast<size_t>(anchor) * KBuild + i + threadIdx.x]
                                   : Cache::EMPTY_KEY;
        }
        cache.fetch(s_knn, nullptr, K_BLOCK);
      }
    }

    __syncthreads();

    // When results from multiple shards are concatenated, we write to a specific slice.
    const size_t result_offset = static_cast<size_t>(n) * KQuery * shards_per_gpu +
                                 static_cast<size_t>(on_gpu_shard_id) * KQuery;
    const uint32_t index_offset = on_gpu_shard_id * N_base;
    cache.write_best(d_query_results + result_offset, KQuery, index_offset);

    if constexpr (WRITE_DISTS) {
      for (uint32_t k = threadIdx.x; k < KQuery; k += BLOCK_DIM_X) {
        d_query_results_dists[result_offset + k] = cache.s_dists[k];
      }
    }

    if constexpr (DIST_STATS) {
      if (!threadIdx.x) {
        d_dist_stats[n] = cache.get_dist_stats();
      }
    }
  }
};

/**
 * __global__ launcher for the QueryKernel
 */
template <typename KernelT>
__global__ void query_kernel_launcher(KernelT kernel)
{
  kernel();
}

}  // namespace detail
}  // namespace cuda
}  // namespace ggnn