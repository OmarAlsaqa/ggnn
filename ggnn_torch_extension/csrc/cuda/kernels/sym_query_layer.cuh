#pragma once

#include "../utils/simple_knn_sym_cache.cuh"
#include "ggnn_cpu.h"

namespace ggnn {
namespace cuda {
namespace detail {

/**
 * Kernel struct for finding and adding symmetric links to diversify the graph.
 */
template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          uint32_t DIST_ITEMS_PER_THREAD>
struct SymQueryKernel {
  // --- Kernel Configuration ---
  static constexpr uint32_t BLOCK_DIM_X = BLOCK_SIZE;
  static constexpr uint32_t MAX_PER_PATH_ITERATIONS = 20;
  static constexpr uint32_t CACHE_SIZE = 128;
  static constexpr uint32_t MIN_PRIOQ_SIZE = 16;
  const uint32_t sorted_size =
      std::max(CACHE_SIZE < 512U ? 64U : 32U,
               ::ggnn::next_multiple<uint32_t, 32>(KBuild / 2 + MIN_PRIOQ_SIZE));

  // --- Member Variables (Kernel Parameters) ---
  uint32_t D;
  DistanceMeasure measure;
  uint32_t KBuild;
  const BaseT* d_base;
  const KeyT* d_graph;
  const KeyT* d_translation;
  const float* d_nn1_stats;
  float tau_build;
  KeyT* d_sym_buffer;
  const uint32_t* d_sym_atomic;

  // --- CONSTRUCTOR ---
  SymQueryKernel(uint32_t D, DistanceMeasure measure, uint32_t KBuild, const BaseT* d_base,
                 const KeyT* d_graph, const KeyT* d_translation, const float* d_nn1_stats,
                 float tau_build, KeyT* d_sym_buffer, const uint32_t* d_sym_atomic)
      : D(D),
        measure(measure),
        KBuild(KBuild),
        d_base(d_base),
        d_graph(d_graph),
        d_translation(d_translation),
        d_nn1_stats(d_nn1_stats),
        tau_build(tau_build),
        d_sym_buffer(d_sym_buffer),
        d_sym_atomic(d_sym_atomic)
  {
  }
  // --- Device-Side Implementation ---
  // (Implementation from original sym_query_layer.cu)
  __device__ __forceinline__ void operator()() const
  {
    static constexpr uint32_t K_BLOCK = 32;
    static_assert(K_BLOCK <= BLOCK_DIM_X);
    static constexpr bool DIST_STATS = false;

    const uint32_t KF = KBuild / 2;
    const uint32_t KL = KBuild - KF;

    using Cache = ::ggnn::cuda::SimpleKNNSymCache<KeyT, ValueT, BaseT, BLOCK_DIM_X,
                                                  DIST_ITEMS_PER_THREAD, DIST_STATS>;

    const float xi = (measure == DistanceMeasure::Euclidean)
                         ? (d_nn1_stats[0] * d_nn1_stats[0]) * tau_build * tau_build
                         : d_nn1_stats[0] * tau_build;

    const KeyT n = static_cast<KeyT>(blockIdx.x);

    Cache cache(D, measure, KF, sorted_size, CACHE_SIZE, d_base,
                d_translation ? d_translation[n] : n, xi);

    __shared__ bool s_connected;

    // For each of the primary neighbors of node 'n'...
    for (uint32_t i = 0; i < KL; ++i) {
      const KeyT primary_neighbor_id = d_graph[static_cast<size_t>(n) * KBuild + i];

      if (!threadIdx.x)
        s_connected = false;
      __syncthreads();

      // ...start a search from that neighbor to see if we can find a path back to 'n'.
      cache.init_start_point(primary_neighbor_id, d_translation);

      bool found_connection_in_warp = false;

      for (uint32_t ite = 0; ite < MAX_PER_PATH_ITERATIONS && !s_connected; ++ite) {
        const KeyT anchor = cache.pop();

        if (anchor == Cache::EMPTY_KEY)
          break;

        // Explore the neighbors of the current anchor point.
        __shared__ KeyT s_knn[K_BLOCK];
        for (uint32_t j = 0; j < KBuild; j += K_BLOCK) {
          if (threadIdx.x < K_BLOCK) {
            const uint32_t k = j + threadIdx.x;
            if (k < KBuild) {
              const KeyT neighbor_of_anchor =
                  (k < KL) ? d_graph[static_cast<size_t>(anchor) * KBuild + k]
                           : d_sym_buffer[static_cast<size_t>(anchor) * KF + k - KL];

              if (neighbor_of_anchor == n) {
                s_connected = true;  // Path back to 'n' found!
              }
              s_knn[threadIdx.x] = neighbor_of_anchor;
            }
            else {
              s_knn[threadIdx.x] = Cache::EMPTY_KEY;
            }
          }
          __syncthreads();

          // Check if any thread in the block found the connection
          if (s_connected) {
            found_connection_in_warp = true;
            break;
          }
          cache.fetch(s_knn, d_translation, K_BLOCK);
        }
        if (found_connection_in_warp)
          break;
      }

      __syncthreads();

      // If after the search, no path back to 'n' was found...
      if (!s_connected) {
        // ...we need to add a symmetric link to 'n'.
        if (!threadIdx.x) {
          // Try to add the link to one of the K nearest neighbors found during the path search.
          for (uint32_t best_k = 0; best_k < KF; ++best_k) {
            const KeyT candidate_node = cache.s_cache[best_k];
            if (candidate_node == Cache::EMPTY_KEY)
              break;

            // Atomically increment the counter for this candidate node.
            const uint32_t pos = atomicAdd(const_cast<uint32_t*>(&d_sym_atomic[candidate_node]), 1U);

            // If we got a slot, write our ID ('n') into it and stop.
            if (pos < KF) {
              d_sym_buffer[static_cast<size_t>(candidate_node) * KF + pos] = n;
              break;
            }
          }
        }
      }
    }
  }
};

/**
 * __global__ launcher for the SymQueryKernel
 */
template <typename KernelT>
__global__ void sym_query_kernel_launcher(KernelT kernel)
{
  kernel();
}

}  // namespace detail
}  // namespace cuda
}  // namespace ggnn