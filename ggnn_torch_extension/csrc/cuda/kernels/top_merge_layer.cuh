#pragma once

#include "../utils/distance.cuh"
#include "../utils/k_best_list.cuh"
#include "../../ggnn_config.h"
#include "../../cpu/ggnn_cpu.h"

namespace ggnn {
namespace cuda {
namespace detail {

/**
 * Kernel struct for the initial brute-force k-NN search within a data segment.
 * This is the first step of the hierarchical build.
 */
template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          uint32_t DIST_ITEMS_PER_THREAD>
struct TopMergeKernel {
  // --- Kernel Configuration ---
  static constexpr uint32_t BLOCK_DIM_X = BLOCK_SIZE;

  // --- Member Variables (Kernel Parameters) ---
  uint32_t D;
  DistanceMeasure measure;
  uint32_t KBuild;
  const BaseT* d_base;
  KeyT* d_translation;
  KeyT* d_graph;
  ValueT* d_nn1_dist_buffer;
  uint32_t S;
  uint32_t S_offset;
  uint32_t layer;

  // --- CONSTRUCTOR ---
  TopMergeKernel(uint32_t D_in, DistanceMeasure measure_in, uint32_t KBuild_in,
                 const BaseT* d_base_in, KeyT* d_translation_in, KeyT* d_graph_in,
                 ValueT* d_nn1_dist_buffer_in, uint32_t S_in, uint32_t S_offset_in,
                 uint32_t layer_in)
      : D(D_in),
        measure(measure_in),
        KBuild(KBuild_in),
        d_base(d_base_in),
        d_translation(d_translation_in),
        d_graph(d_graph_in),
        d_nn1_dist_buffer(d_nn1_dist_buffer_in),
        S(S_in),
        S_offset(S_offset_in),
        layer(layer_in)
  {
  }

  // --- Device-Side Implementation ---
  // (Implementation from original top_merge_layer.cu)
  __device__ void operator()()
  {
    using Distance = ggnn::cuda::Distance<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD>;
    using KBestList = ggnn::cuda::KBestList<KeyT, ValueT, BLOCK_DIM_X>;

    // Each thread block is responsible for finding the neighbors of one node.
    const uint32_t n = blockIdx.x;
    const KeyT m = (!layer) ? n : d_translation[n];

    Distance distCalc(D, measure, d_base, m);
    KBestList best(KBuild);

    // Determine the start and end of the segment this node belongs to.
    const uint32_t S_plus_offset = S_offset * (S + 1);
    const uint32_t S_actual = (!layer && n < S_plus_offset) ? S + 1 : S;
    const KeyT start = (layer || n < S_plus_offset)
                           ? (n / S_actual) * S_actual
                           : S_plus_offset + ((n - S_plus_offset) / S_actual) * S_actual;
    const KeyT end = start + S_actual;

    // Iterate through all other points in the same segment.
    for (KeyT other_n = start; other_n < end; ++other_n) {
      const KeyT other_m = (layer) ? d_translation[other_n] : other_n;

      if (m == other_m) {
        continue;  // Don't compare a point to itself
      }

      ValueT dist = distCalc.distance_synced(other_m);
      best.add_unique(dist, other_n);
    }
    __syncthreads();

    // Write the K best neighbors found back to global memory.
    for (uint32_t k = threadIdx.x; k < KBuild; k += BLOCK_DIM_X) {
      d_graph[static_cast<size_t>(n) * KBuild + k] = best.s_ids[k];
    }

    // Thread 0 also writes the distance to the single nearest neighbor.
    if (!threadIdx.x) {
      ValueT nn1_dist = best.s_dists[0];  // The list is sorted, so index 0 is the best
      if (measure == DistanceMeasure::Euclidean) {
        nn1_dist = sqrtf(nn1_dist);
      }
      d_nn1_dist_buffer[n] = nn1_dist;
    }
  }
};

/**
 * __global__ launcher for the TopMergeKernel
 */
template <typename KernelT>
__global__ void top_merge_kernel_launcher(KernelT kernel)
{
  kernel();
}

}  // namespace detail
}  // namespace cuda
}  // namespace ggnn