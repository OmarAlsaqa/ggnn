#pragma once

#include "distance.cuh"
#include "../../cpu/ggnn_cpu.h"
#include "../../ggnn_config.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

// Move the utility into the ggnn::cuda namespace for consistency
namespace ggnn {
namespace cuda {

/**
 * @brief A high-performance cache for the GGNN search algorithm, designed
 * to be managed in parallel by a CUDA thread block using shared memory.
 *
 * It maintains three key data structures:
 * 1. A sorted list of the K best candidates found so far.
 * 2. A priority queue (implemented as a ring buffer) of candidates to visit.
 * 3. A "visited" list to prevent cycles during the graph search.
 */
template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_DIM_X,
          uint32_t DIST_ITEMS_PER_THREAD, bool DIST_STATS = false>
struct SimpleKNNCache {
  static constexpr KeyT EMPTY_KEY = static_cast<KeyT>(-1);
  static constexpr ValueT EMPTY_DIST = std::numeric_limits<ValueT>::infinity();

 private:
  // Note: The Distance struct is now in the same ggnn::cuda namespace
  using Distance = ggnn::cuda::Distance<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD>;

 public:
  const uint32_t BEST_SIZE;
  const uint32_t SORTED_SIZE;
  const uint32_t CACHE_SIZE;

  KeyT* s_cache;
  ValueT* s_dists;
  uint32_t r_prioQ_head;
  uint32_t r0_visited_head;

  bool& s_sync;
  Distance rs_dist_calc;
  ValueT r_xi;
  uint32_t r0_dist_calc_counter;

  __device__ __forceinline__ void initSharedStorage()
  {
    extern __shared__ KeyT shared_cache[];
    s_cache = shared_cache;
    s_dists = reinterpret_cast<ValueT*>(&s_cache[CACHE_SIZE]);
  }

  __device__ __forceinline__ bool& SyncPrivateTmpStorage()
  {
    __shared__ bool s_sync_tmp;
    return s_sync_tmp;
  }

  __device__ __forceinline__ void init()
  {
    for (uint32_t i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
      s_cache[i] = EMPTY_KEY;
      if (i < SORTED_SIZE)
        s_dists[i] = EMPTY_DIST;
    }
    r_prioQ_head = BEST_SIZE;
    if (threadIdx.x == 0) {
      if constexpr (DIST_STATS)
        r0_dist_calc_counter = 0;
      r0_visited_head = SORTED_SIZE;
    }
    __syncthreads();
  }

  // Constructor for graph construction searches
  __device__ __forceinline__ SimpleKNNCache(const uint32_t D, const DistanceMeasure measure,
                                            const uint32_t BEST_SIZE, const uint32_t SORTED_SIZE,
                                            const uint32_t CACHE_SIZE, const BaseT* d_base,
                                            const KeyT n, const ValueT xi_criteria)
      : BEST_SIZE{BEST_SIZE},
        SORTED_SIZE{SORTED_SIZE},
        CACHE_SIZE(CACHE_SIZE),
        s_sync(SyncPrivateTmpStorage()),
        rs_dist_calc(D, measure, d_base, n),
        r_xi(xi_criteria)
  {
    initSharedStorage();
    init();
  }

  // Constructor for external queries
  __device__ __forceinline__ SimpleKNNCache(const uint32_t D, const DistanceMeasure measure,
                                            const uint32_t BEST_SIZE, const uint32_t SORTED_SIZE,
                                            const uint32_t CACHE_SIZE, const BaseT* d_base,
                                            const BaseT* d_query, const KeyT n,
                                            const ValueT xi_criteria)
      : BEST_SIZE{BEST_SIZE},
        SORTED_SIZE{SORTED_SIZE},
        CACHE_SIZE(CACHE_SIZE),
        s_sync(SyncPrivateTmpStorage()),
        rs_dist_calc(D, measure, d_base, d_query, n),
        r_xi(xi_criteria)
  {
    initSharedStorage();
    init();
  }

  __device__ __forceinline__ ValueT criteria() const
  {
    return s_dists[BEST_SIZE - 1] + r_xi;
  }

  __device__ __forceinline__ void push(const KeyT key, const ValueT dist)
  {
    // Check for duplicates in the entire cache (best list, prioQ, and visited list)
    if (threadIdx.x == 0)
      s_sync = false;
    __syncthreads();

    for (uint32_t idx = threadIdx.x; idx < r0_visited_head && !s_sync; idx += BLOCK_DIM_X) {
      if (s_cache[idx] == key) {
        s_sync = true;
      }
    }
    __syncthreads();
    if (s_sync)
      return;

    const uint32_t head_idx_prioQ = r_prioQ_head;

    // Parallel insertion sort logic...
    // This is a complex but highly optimized piece of code.
    KeyT r_cache;
    ValueT r_dists;
    uint32_t idx;
    bool active = false;
    uint32_t block_start = ((SORTED_SIZE + BLOCK_DIM_X - 1) / BLOCK_DIM_X) * BLOCK_DIM_X;

    while (true) {
      if (active) {
        if (r_cache != EMPTY_KEY) {
          const uint32_t idx_next = (idx + 1 == SORTED_SIZE) ? BEST_SIZE : idx + 1;
          if (idx_next < head_idx_prioQ ||
              (head_idx_prioQ == BEST_SIZE && idx_next < SORTED_SIZE)) {
            s_cache[idx_next] = r_cache;
            s_dists[idx_next] = r_dists;
          }
        }
        const bool has_prev = idx != 0 && idx != head_idx_prioQ;
        const uint32_t idx_prev = (idx == BEST_SIZE) ? SORTED_SIZE - 1 : idx - 1;
        if (!has_prev || s_dists[idx_prev] <= dist) {
          s_cache[idx] = key;
          s_dists[idx] = dist;
        }
      }
      if (block_start == 0)
        break;
      block_start -= BLOCK_DIM_X;
      idx = block_start + threadIdx.x;
      active = idx < SORTED_SIZE;
      if (active) {
        const uint32_t head_idx_in_prioQ = head_idx_prioQ - BEST_SIZE;
        if (idx >= BEST_SIZE) {
          idx = (idx - BEST_SIZE + head_idx_in_prioQ < SORTED_SIZE - BEST_SIZE)
                    ? idx + head_idx_in_prioQ
                    : idx + head_idx_in_prioQ - (SORTED_SIZE - BEST_SIZE);
        }
        r_cache = s_cache[idx];
        r_dists = s_dists[idx];
        active &= (dist < r_dists);
      }
      __syncthreads();
    }
  }

  __device__ __forceinline__ KeyT pop()
  {
    __syncthreads();
    const uint32_t head_idx_prioQ = r_prioQ_head;
    const KeyT key = s_cache[head_idx_prioQ];
    const ValueT dist = s_dists[head_idx_prioQ];
    __syncthreads();

    if (key == EMPTY_KEY || dist >= criteria()) {
      return EMPTY_KEY;
    }

    if (threadIdx.x == 0) {
      const uint32_t head_idx_visited = r0_visited_head;
      s_cache[head_idx_visited] = key;
      r0_visited_head = (head_idx_visited + 1) >= CACHE_SIZE ? SORTED_SIZE : head_idx_visited + 1;
      s_cache[head_idx_prioQ] = EMPTY_KEY;
      s_dists[head_idx_prioQ] = EMPTY_DIST;
    }

    r_prioQ_head = (head_idx_prioQ + 1) >= SORTED_SIZE ? BEST_SIZE : head_idx_prioQ + 1;
    __syncthreads();
    return key;
  }

  template <bool filter_known_keys = true>
  __device__ __forceinline__ void fetch(
      std::conditional_t<filter_known_keys, KeyT*, const KeyT*> s_keys, const KeyT* d_translation,
      uint32_t len)
  {
    if constexpr (filter_known_keys) {
      __syncthreads();
      // Filter out keys that are already in our cache
      for (uint32_t i = threadIdx.x; i < r0_visited_head; i += BLOCK_DIM_X) {
        const KeyT n = s_cache[i];
        if (n == EMPTY_KEY)
          continue;
        for (uint32_t k = 0; k < len; ++k) {
          if (s_keys[k] == n) {
            s_keys[k] = EMPTY_KEY;
          }
        }
      }
    }
    __syncthreads();

    // Each thread processes candidates from the s_keys list
    for (uint32_t k = threadIdx.x; k < len; k += BLOCK_DIM_X) {
      const KeyT other_n = s_keys[k];
      if (other_n == EMPTY_KEY)
        continue;

      const KeyT other_m = (d_translation) ? d_translation[other_n] : other_n;
      const ValueT dist = rs_dist_calc.distance_synced(other_m);

      if (dist < criteria()) {
        push(other_n, dist);
      }
    }
    __syncthreads();
  }

  __device__ __forceinline__ void fetch_unfiltered(const KeyT* s_keys, const KeyT* d_translation,
                                                   const uint32_t len)
  {
    fetch<false>(s_keys, d_translation, len);
  }

  __device__ __forceinline__ void transform(const KeyT* transform)
  {
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
      if (i < BEST_SIZE) {
        KeyT key = s_cache[i];
        if (key != EMPTY_KEY)
          s_cache[i] = transform[key];
        if (i + BEST_SIZE < SORTED_SIZE) {
          s_cache[i + BEST_SIZE] = s_cache[i];
          s_dists[i + BEST_SIZE] = s_dists[i];
        }
      }
      else if (i >= 2 * BEST_SIZE || i >= SORTED_SIZE) {
        s_cache[i] = EMPTY_KEY;
        if (i < SORTED_SIZE)
          s_dists[i] = EMPTY_DIST;
      }
    }
    r_prioQ_head = BEST_SIZE;
    if (threadIdx.x == 0) {
      r0_visited_head = SORTED_SIZE;
    }
    __syncthreads();
  }

  __device__ __forceinline__ void write_best(KeyT* d_buffer, uint32_t K, uint32_t idx_offset)
  {
    for (uint32_t i = threadIdx.x; i < K; i += BLOCK_DIM_X) {
      if (i < BEST_SIZE) {
        const KeyT idx = s_cache[i];
        d_buffer[i] = (idx != EMPTY_KEY) ? (idx + idx_offset) : EMPTY_KEY;
      }
      else {
        d_buffer[i] = EMPTY_KEY;
      }
    }
  }

  __device__ __forceinline__ uint32_t get_dist_stats()
  {
    return r0_dist_calc_counter;
  }
};

}  // namespace cuda
}  // namespace ggnn