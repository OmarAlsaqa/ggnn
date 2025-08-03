#pragma once

#include "../utils/simple_knn_cache.cuh"
#include "../../ggnn_config.h"
#include "../../cpu/ggnn_cpu.h"

namespace ggnn {
namespace cuda {
namespace detail {

/**
 * Kernel struct for the hierarchical graph merge operation.
 */
template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          uint32_t DIST_ITEMS_PER_THREAD>
struct MergeKernel {
  // --- Kernel Configuration ---
  static constexpr uint32_t BLOCK_DIM_X = BLOCK_SIZE;
  static constexpr uint32_t MAX_ITERATIONS = 200;
  static constexpr uint32_t CACHE_SIZE = 256;
  static constexpr uint32_t MIN_PRIOQ_SIZE = 16;
  const uint32_t SORTED_SIZE =
      std::max(CACHE_SIZE < 512U ? 64U : 32U,
               ::ggnn::next_multiple<uint32_t, 32>(KBuild + 1 + MIN_PRIOQ_SIZE));

  // --- Member Variables (Kernel Parameters) ---
  uint32_t D;
  DistanceMeasure measure;
  uint32_t KBuild;
  uint32_t S;
  const BaseT* d_base;
  KeyT* d_selection;
  KeyT* d_translation;
  KeyT* d_graph;
  KeyT* d_graph_buffer;
  const float* d_nn1_stats;
  float* d_nn1_dist_buffer;
  uint32_t layer_top;
  uint32_t layer_btm;
  uint32_t G;
  uint32_t S0;
  uint32_t S0_offset;
  std::array<uint32_t, 4> Ns_offsets;
  std::array<uint32_t, 4> STs_offsets;
  float tau_build;

  // --- CONSTRUCTOR ---
  MergeKernel(uint32_t D_in, DistanceMeasure measure_in, uint32_t KBuild_in, const BaseT* d_base_in,
              KeyT* d_selection_in, KeyT* d_translation_in, KeyT* d_graph_in,
              KeyT* d_graph_buffer_in, const float* d_nn1_stats_in, float* d_nn1_dist_buffer_in,
              uint32_t S_in, uint32_t layer_top_in, uint32_t layer_btm_in, uint32_t G_in,
              uint32_t S0_in, uint32_t S0_offset_in, const std::array<uint32_t, 4>& Ns_offsets_in,
              const std::array<uint32_t, 4>& STs_offsets_in, float tau_build_in)
      : D(D_in),
        measure(measure_in),
        KBuild(KBuild_in),
        S(S_in),
        d_base(d_base_in),
        // The selection pointer is derived from the translation pointer
        d_selection(d_translation_in - STs_offsets_in[layer_top_in] + STs_offsets_in[layer_btm_in]),
        d_translation(d_translation_in),
        d_graph(d_graph_in),
        d_graph_buffer(d_graph_buffer_in),
        d_nn1_stats(d_nn1_stats_in),
        d_nn1_dist_buffer(d_nn1_dist_buffer_in),
        layer_top(layer_top_in),
        layer_btm(layer_btm_in),
        G(G_in),
        S0(S0_in),
        S0_offset(S0_offset_in),
        Ns_offsets(Ns_offsets_in),
        STs_offsets(STs_offsets_in),
        tau_build(tau_build_in)
  {
  }
  // --- Device-Side Helper Function ---
  // (Implementation from original merge_layer.cu)
  __device__ __forceinline__ uint32_t get_top_seg_offset(const KeyT n) const
  {
    // first, determine the bottom-level segment
    uint32_t seg_btm = n / S;
    if (!layer_btm) {
      const KeyT offset_points = S0_offset * (S0 + 1);
      seg_btm = (n < offset_points) ? n / (S0 + 1) : S0_offset + (n - offset_points) / S0;
    }

    // then divide by G once per layer to step up the tree
    // and finally multiply by S to get the start of the segment
    uint32_t powG = G;
    for (uint32_t i = 1; i < layer_top - layer_btm; ++i)
      powG *= G;

    return (seg_btm / powG) * S;
  }

  // --- Device-Side Implementation ---
  // (Implementation from original merge_layer.cu)
  __device__ __forceinline__ void operator()()
  {
    static constexpr uint32_t K_BLOCK = 32;
    static_assert(K_BLOCK <= BLOCK_DIM_X);
    static constexpr bool DIST_STATS = false;

    using Cache = ::ggnn::cuda::SimpleKNNCache<KeyT, ValueT, BaseT, BLOCK_DIM_X,
                                               DIST_ITEMS_PER_THREAD, DIST_STATS>;

    const float xi = (measure == DistanceMeasure::Euclidean)
                         ? (d_nn1_stats[0] * d_nn1_stats[0]) * tau_build * tau_build
                         : d_nn1_stats[0] * tau_build;

    const KeyT n = static_cast<KeyT>(blockIdx.x);
    const KeyT m = (!layer_btm) ? n : d_translation[STs_offsets[layer_btm] + n];

    Cache cache(D, measure, KBuild + 1, SORTED_SIZE, CACHE_SIZE, d_base, m, xi);

    __shared__ KeyT s_knn[K_BLOCK];

    {
      const uint32_t s_offset = get_top_seg_offset(n);
      // fetch starting points
      for (uint32_t i = 0; i < S; i += K_BLOCK) {
        if (threadIdx.x < K_BLOCK) {
          const uint32_t s = i + threadIdx.x;
          s_knn[threadIdx.x] = (s < S) ? static_cast<KeyT>(s_offset + s) : Cache::EMPTY_KEY;
        }
        cache.fetch_unfiltered(s_knn, &d_translation[STs_offsets[layer_top]], K_BLOCK);
      }
    }

    // hierarchic kNN search
    for (uint32_t layer = layer_top - 1; layer >= layer_btm && layer != -1U; layer--) {
      cache.transform(&d_selection[STs_offsets[layer + 1]]);

      if (layer == layer_btm) {
        cache.fetch_unfiltered(&n, (!layer) ? nullptr : &d_translation[STs_offsets[layer]], 1);
      }

      for (uint32_t ite = 0; ite < MAX_ITERATIONS; ++ite) {
        const KeyT anchor = cache.pop();
        if (anchor == Cache::EMPTY_KEY)
          break;

        for (uint32_t j = 0; j < KBuild; j += K_BLOCK) {
          if (threadIdx.x < K_BLOCK) {
            const uint32_t k = j + threadIdx.x;
            s_knn[threadIdx.x] =
                (k < KBuild)
                    ? d_graph[(static_cast<size_t>(Ns_offsets[layer]) + anchor) * KBuild + k]
                    : Cache::EMPTY_KEY;
          }
          cache.fetch(s_knn, (!layer) ? nullptr : &d_translation[STs_offsets[layer]], K_BLOCK);
        }
      }
    }

    KeyT s_own_idx;  // Use a register, not shared memory for this
    if (!threadIdx.x)
      s_knn[0] = static_cast<KeyT>(-1);  // Re-use shared mem for communication
    __syncthreads();

    // Check if own index is part of cache and mark its index to skip it
    for (uint32_t j = 0; j < KBuild + 1; j += BLOCK_DIM_X) {
      const uint32_t k = j + threadIdx.x;
      if (k < KBuild + 1) {
        if (cache.s_cache[k] == n) {
          atomicMax((unsigned int*)&s_knn[0], (unsigned int)k);
        }
      }
    }
    __syncthreads();
    s_own_idx = s_knn[0];

    for (uint32_t j = 0; j < KBuild; j += BLOCK_DIM_X) {
      const uint32_t k = j + threadIdx.x;
      if (k < KBuild) {
        const KeyT idx = cache.s_cache[k + (s_own_idx != static_cast<KeyT>(-1) && k >= s_own_idx)];
        d_graph_buffer[static_cast<size_t>(n) * KBuild + k] = (idx != Cache::EMPTY_KEY) ? idx : n;
      }
    }

    if (!layer_btm && !threadIdx.x) {
      ValueT dist = std::numeric_limits<ValueT>::infinity();
      for (uint32_t i = 0; i < KBuild + 1; ++i) {
        if (cache.s_cache[i] != n) {
          dist = cache.s_dists[i];
          break;
        }
      }
      if (measure == DistanceMeasure::Euclidean) {
        dist = sqrtf(dist);
      }
      d_nn1_dist_buffer[n] = dist;
    }
  }
};

/**
 * __global__ launcher for the MergeKernel
 */
template <typename KernelT>
__global__ void merge_kernel_launcher(KernelT kernel)
{
  kernel();
}

}  // namespace detail
}  // namespace cuda
}  // namespace ggnn