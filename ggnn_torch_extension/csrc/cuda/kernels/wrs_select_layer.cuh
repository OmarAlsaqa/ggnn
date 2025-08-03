#pragma once

#include <cub/cub.cuh>
#include <limits>
#include "ggnn_cpu.h"

namespace ggnn {
namespace cuda {
namespace detail {

/**
 * Kernel struct for Weighted Random Sampling (WRS) to select points
 * for the next hierarchical graph layer.
 */
template <typename KeyT, typename ValueT>
struct WRSSelectionKernel {
  // --- Kernel Configuration ---
  static constexpr uint32_t BLOCK_DIM_X = 128;
  static constexpr uint32_t ITEMS_PER_THREAD = 2;

  // --- Member Variables (Kernel Parameters) ---
  const KeyT* d_selection;
  const KeyT* d_translation;
  const KeyT* d_translation_layer;
  float* d_nn1_dist_buffer;
  const float* d_rng;
  uint32_t Sglob;
  uint32_t S;
  uint32_t S_offset;
  uint32_t G;
  uint32_t SG;
  uint32_t SG_offset;
  uint32_t layer;

  // --- CONSTRUCTOR ---
  WRSSelectionKernel(const KeyT* d_selection, const KeyT* d_translation,
                     const KeyT* d_translation_layer, float* d_nn1_dist_buffer, const float* d_rng,
                     uint32_t Sglob, uint32_t S, uint32_t S_offset, uint32_t G, uint32_t SG,
                     uint32_t SG_offset, uint32_t layer)
      : d_selection(d_selection),
        d_translation(d_translation),
        d_translation_layer(d_translation_layer),
        d_nn1_dist_buffer(d_nn1_dist_buffer),
        d_rng(d_rng),
        Sglob(Sglob),
        S(S),
        S_offset(S_offset),
        G(G),
        SG(SG),
        SG_offset(SG_offset),
        layer(layer)
  {
  }

  // --- Device-Side Implementation ---
  // (Implementation from original wrs_select_layer.cu)
  __device__ void operator()() // Removed 'const'
  {
    using BlockRadixSort = cub::BlockRadixSort<ValueT, BLOCK_DIM_X, ITEMS_PER_THREAD, KeyT>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Each thread block processes one segment of the current layer.
    const uint32_t b = blockIdx.x;

    // Calculate the actual size and starting point of this segment.
    const uint32_t S_current = S + (b < S_offset);
    const uint32_t start = b * S + std::min(b, S_offset);

    ValueT keys[ITEMS_PER_THREAD];  // Weights for sorting
    KeyT values[ITEMS_PER_THREAD];  // Point indices

    // Each thread loads ITEMS_PER_THREAD points from the segment.
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
      const uint32_t i = item * BLOCK_DIM_X + threadIdx.x;
      if (i < S_current) {
        const KeyT n = start + i;
        // Calculate the weight using the formula from the paper.
        const float weight = (-1.0f * logf(d_rng[n] + 1e-9f)) /
                             (d_nn1_dist_buffer[n] + std::numeric_limits<float>::epsilon());
        keys[item] = weight;
        values[item] = n;
      }
      else {
        // Pad with invalid entries if the segment is smaller than the block's capacity.
        keys[item] = -1.0f;
        values[item] = -1;
      }
    }

    // Use CUB's fast parallel radix sort within the thread block.
    // We sort in descending order to find the points with the highest weights.
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(keys, values);

    // Determine how many points this segment should contribute to the next layer.
    const uint32_t upper_segment = b / G;
    const uint32_t nth_lower_segment = b - upper_segment * G;
    const uint32_t num_selected_points = SG + (nth_lower_segment < SG_offset);

    // Calculate the destination offset in the upper layer's selection/translation arrays.
    const uint32_t dest =
        upper_segment * Sglob + nth_lower_segment * SG + std::min(nth_lower_segment, SG_offset);

    __syncthreads();

    // Each thread writes out its assigned points from the top of the sorted list.
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
      const uint32_t s = threadIdx.x + item * BLOCK_DIM_X;
      if (s < num_selected_points) {
        const KeyT n = values[item];  // Get the ID of the selected point.

        // Write the selected point's ID to the selection array.
        d_selection[dest + s] = n;
        // Write the point's translated ID (from the base layer) to the translation array.
        d_translation[dest + s] = (!layer) ? n : d_translation_layer[n];
      }
    }
  }
};

/**
 * __global__ launcher for the WRSSelectionKernel
 */
template <typename KernelT>
__global__ void select_kernel_launcher(KernelT kernel)
{
  kernel();
}

}  // namespace detail
}  // namespace cuda
}  // namespace ggnn