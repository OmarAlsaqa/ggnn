#pragma once

#include "../../cpu/ggnn_cpu.h"
#include "../../ggnn_config.h"

namespace ggnn {
namespace cuda {
namespace detail {

/**
 * Kernel struct for merging newly found symmetric links from a temporary
 * buffer back into the main graph structure.
 */
template <typename KeyT, typename ValueT>
struct SymBufferMergeKernel {
  // --- Kernel Configuration ---
  static constexpr uint32_t BLOCK_DIM_X = 128;  // This is the total number of threads per block

  // --- Member Variables (Kernel Parameters) ---
  uint32_t KBuild;
  uint32_t KF{KBuild / 2};                      // Number of foreign/symmetric links
  uint32_t KL{KBuild - KF};                     // Number of local/primary links
  uint32_t POINTS_PER_BLOCK{BLOCK_DIM_X / KF};  // How many graph nodes each thread block handles

  KeyT* d_sym_buffer;
  uint32_t* d_sym_atomic;
  KeyT* d_graph;

  // --- CONSTRUCTOR ---
  SymBufferMergeKernel(uint32_t KBuild, KeyT* d_sym_buffer, uint32_t* d_sym_atomic,
                       KeyT* d_graph)
      : KBuild(KBuild),
        KF(KBuild / 2),
        KL(KBuild - KF),
        POINTS_PER_BLOCK(BLOCK_DIM_X / KF),
        d_sym_buffer(d_sym_buffer),
        d_sym_atomic(d_sym_atomic),
        d_graph(d_graph)
  {
  }

  // --- Device-Side Implementation ---
  // (Implementation from original sym_buffer_merge_layer.cu)
  __device__ void operator()(uint32_t N) 
  {
    // This kernel uses a 2D thread indexing scheme for convenience
    const uint32_t n = blockIdx.x * POINTS_PER_BLOCK + threadIdx.y;
    const uint32_t kf = threadIdx.x;  // Thread's responsibility within the foreign link list

    if (n >= N) {
      return;
    }

    // Use shared memory to stage data and avoid repeated global memory access
    extern __shared__ KeyT s_sym_buffer[];
    KeyT* s_graph_foreign_links = &s_sym_buffer[POINTS_PER_BLOCK * KF];
    bool* s_is_found = reinterpret_cast<bool*>(&s_graph_foreign_links[POINTS_PER_BLOCK * KF]);

    // Each thread in the warp handles one potential foreign link
    const uint32_t tid_1d = threadIdx.y * KF + threadIdx.x;

    // Stage 1: Load data from global to shared memory
    s_sym_buffer[tid_1d] = d_sym_buffer[static_cast<size_t>(n) * KF + kf];
    s_graph_foreign_links[tid_1d] = d_graph[static_cast<size_t>(n) * KBuild + KL + kf];

    uint32_t num_new_links;
    if (threadIdx.x == 0) {
      num_new_links = d_sym_atomic[n];
    }
    __syncthreads();

    // Stage 2: Merge existing links with new links
    // We want to keep the existing links if they aren't already in the new `sym_buffer` list.
    for (uint32_t i = 0; i < KF; ++i) {
      if (threadIdx.x == 0) {
        // One thread per node checks if its existing link `i` is already present
        // in the list of new links found by the sym_query kernel.
        s_is_found[threadIdx.y] = false;
        if (num_new_links >= KF) {
          s_is_found[threadIdx.y] = true;  // Buffer is full, can't add existing links
        }
        else {
          KeyT existing_link = s_graph_foreign_links[threadIdx.y * KF + i];
          for (uint32_t j = 0; j < num_new_links; ++j) {
            if (existing_link == s_sym_buffer[threadIdx.y * KF + j]) {
              s_is_found[threadIdx.y] = true;
              break;
            }
          }
        }
      }
      __syncthreads();

      // If the existing link was not found and there's space, add it to the buffer.
      if (threadIdx.x == 0 && !s_is_found[threadIdx.y]) {
        KeyT existing_link = s_graph_foreign_links[threadIdx.y * KF + i];
        if (existing_link != -1) {  // Don't add invalid links
          s_sym_buffer[threadIdx.y * KF + num_new_links] = existing_link;
          num_new_links++;
        }
      }
      __syncthreads();
    }

    // Stage 3: Write the merged list back to global memory
    const KeyT res = s_sym_buffer[tid_1d];
    d_graph[static_cast<size_t>(n) * KBuild + KL + kf] = (res >= 0) ? res : n;
  }
};

/**
 * __global__ launcher for the SymBufferMergeKernel
 */
template <typename KernelT>
__global__ void sym_buffer_merge_kernel_launcher(KernelT kernel, const uint32_t N)
{
  kernel(N);
}

}  // namespace detail
}  // namespace cuda
}  // namespace ggnn