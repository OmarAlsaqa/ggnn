#pragma once

#include <cstdint>
#include <limits>
#include <cstdio> // For printf in the debug print function

// Move the utility into the ggnn::cuda namespace for consistency
namespace ggnn {
namespace cuda {

/**
 * KBestList stores the K best elements in parallel using shared memory.
 * This is a device-side utility meant to be used within a CUDA kernel.
 */
template <typename KeyT, typename ValueT, uint32_t BLOCK_DIM_X>
struct KBestList {
  const uint32_t BEST_SIZE;

  ValueT* s_dists;
  KeyT* s_ids;

  static constexpr KeyT EMPTY_KEY = -1;

  __device__ __forceinline__ void initSharedStorage(uint32_t BEST_SIZE)
  {
    // This expects the calling kernel to allocate sufficient shared memory.
    extern __shared__ ValueT shared_kBestList[];
    s_dists = shared_kBestList;
    s_ids = reinterpret_cast<KeyT*>(&s_dists[BEST_SIZE]);
  }

  __device__ __forceinline__ void init()
  {
    for (uint32_t i = threadIdx.x; i < BEST_SIZE; i += BLOCK_DIM_X) {
      s_dists[i] = std::numeric_limits<ValueT>::infinity();
      s_ids[i] = EMPTY_KEY;
    }
    __syncthreads();
  }

  __device__ __forceinline__ KBestList(uint32_t BEST_SIZE) : BEST_SIZE(BEST_SIZE)
  {
    initSharedStorage(BEST_SIZE);
    init();
  }

  __device__ __forceinline__ ValueT worst()
  {
    return s_dists[BEST_SIZE - 1];
  }

  /**
   * Enters an element into the sorted list in parallel.
   * Note: A __syncthreads() is needed after all threads in a block
   * have called this function before the list can be safely read again.
   */
  __device__ __forceinline__ void add_unique(ValueT dist, KeyT id)
  {
    // Process in reverse to simplify shifting logic
    for (int32_t i = ((BEST_SIZE - 1) / BLOCK_DIM_X) * BLOCK_DIM_X; i >= 0; i -= BLOCK_DIM_X) {
      const uint32_t k = i + threadIdx.x;
      ValueT r_dist;
      KeyT r_id;

      if (k < BEST_SIZE) {
        r_dist = s_dists[k];
        r_id = s_ids[k];
      }
      __syncthreads();

      if (k < BEST_SIZE) {
        // If the new distance is smaller than the current value at this position...
        if (dist < r_dist) {
          // ...shift the current value to the right.
          if (k < (BEST_SIZE - 1)) {
            s_dists[k + 1] = r_dist;
            s_ids[k + 1] = r_id;
          }

          // If the position to the left is smaller (or doesn't exist),
          // this is the correct insertion spot.
          if (k == 0 || s_dists[k - 1] <= dist) {
            s_dists[k] = dist;
            s_ids[k] = id;
          }
        }
      }
      __syncthreads();
    }
  }

  /**
   * Transforms all ids w.r.t. a transformation list.
   */
  __device__ __forceinline__ void transform(const KeyT* transform)
  {
    for (uint32_t i = threadIdx.x; i < BEST_SIZE; i += BLOCK_DIM_X) {
      const KeyT id = s_ids[i];
      if (id != EMPTY_KEY) {
        s_ids[i] = transform[id];
      }
    }
  }

  __device__ __forceinline__ void print(int len = -1)
  {
    __syncthreads();
    if (threadIdx.x == 0) {
      printf("KBestList (size %d): \n", BEST_SIZE);
      for (int i = 0; i < BEST_SIZE && (len < 0 || i < len); i++) {
        printf("  [%d]: ID=%d, Dist=%f\n", i, s_ids[i], s_dists[i]);
      }
    }
  }
};

} // namespace cuda
} // namespace ggnn