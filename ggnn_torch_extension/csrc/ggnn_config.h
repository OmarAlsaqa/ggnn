#ifndef GGNN_CONFIG_H
#define GGNN_CONFIG_H

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace ggnn {

// Common enum
enum class DistanceMeasure : int {
  Euclidean = 0,
  Cosine = 1
};
static constexpr uint32_t MAX_NUM_LAYERS = 4;

// Common helper functions
template <typename T, T base>
constexpr T next_multiple(T v) noexcept
{
  return v % base == 0 ? v : base * (v / base + 1);
}
inline size_t align8(size_t size)
{
  return ((size + 7) / 8) * 8;
}
#if __cplusplus < 202002L
template <typename T>
constexpr T bit_ceil(T v) noexcept
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}
#else
using std::bit_ceil;
#endif

/**
 * User-definable graph parameters
 */
struct GraphParameters {
  /// number of base points per shard
  uint32_t N{};
  /// number of dimensions in the dataset and query
  uint32_t D{};

  /// number of neighbors per point
  uint32_t KBuild{};

  /// directory for loading/storing graph shards
  std::filesystem::path graph_dir{};
};

/**
 * Automatically derived secondary graph parameters
 */
struct GraphDerivedParameters : public GraphParameters {
  GraphDerivedParameters() = default;
  GraphDerivedParameters(const GraphParameters& params);

  /// number of inverse (foreign) links per point, part of KBuild
  uint32_t KF{KBuild / 2};

  /// number of layers
  static constexpr uint32_t L{MAX_NUM_LAYERS};

  /// growth factor (number of sub-graphs merged together per layer)
  uint32_t G{};

  /// segment size
  uint32_t S{next_multiple<uint32_t, 32U>(KF + 1)};
  /// segment size in base layer
  uint32_t S0{};
  /// number of segments in base layer with one additional element
  uint32_t S0_off{};

  /// number of points per segment selected into upper-level segment
  uint32_t SG{};
  /// number of segments per layer contributing an additional point into the upper-level segment
  uint32_t SG_off{};
};

/**
 * Automatically derived graph dimensions
 */
struct GraphDimensions {
  GraphDimensions() = default;
  GraphDimensions(const GraphDerivedParameters& params);

  /// total number of neighborhoods in the graph
  uint32_t N_all{};
  /// total number of selection/translation entries
  uint32_t ST_all{};

  /// blocks/segments per layer
  std::array<uint32_t, MAX_NUM_LAYERS> Bs{};  // [L]
  /// neighborhoods per layer
  std::array<uint32_t, MAX_NUM_LAYERS> Ns{};  // [L]
  /// start of neighborhoods per layer
  std::array<uint32_t, MAX_NUM_LAYERS> Ns_offsets{};  // [L]
  /// start of selection/translation per layer
  std::array<uint32_t, MAX_NUM_LAYERS> STs_offsets{};  // [L]
};
/**
 * Combined Configuration of the GGNN search graph layout
 */
struct GraphConfig : public GraphDerivedParameters, public GraphDimensions {
  // Parameters
  uint32_t N;
  uint32_t D;
  uint32_t KBuild;
  // Derived
  uint32_t KF;
  static constexpr uint32_t L{MAX_NUM_LAYERS};
  uint32_t G;
  uint32_t S;
  uint32_t S0;
  uint32_t S0_off;
  uint32_t SG;
  uint32_t SG_off;
  // Dimensions
  uint32_t N_all;
  uint32_t ST_all;
  std::array<uint32_t, MAX_NUM_LAYERS> Bs;
  std::array<uint32_t, MAX_NUM_LAYERS> Ns;
  std::array<uint32_t, MAX_NUM_LAYERS> Ns_offsets;
  std::array<uint32_t, MAX_NUM_LAYERS> STs_offsets;

  GraphConfig(uint32_t n_in, uint32_t d_in, uint32_t k_build_in);
};

namespace cuda {
using KeyT = int32_t;
using ValueT = float;
}  // namespace cuda

// PartSizes structs, also needed by both host and device code
struct GraphPartSizes {
  GraphPartSizes(const GraphConfig& config);
  const size_t graph_size;
  const size_t selection_translation_size;
  const size_t nn1_stats_size;
  size_t getGraphSize() const;
};

struct GraphBufferPartSizes {
  GraphBufferPartSizes(const GraphConfig& config);
  const size_t graph_buffer_size;
  const size_t nn1_dist_buffer_size;
  const size_t rng_size;
  const size_t sym_buffer_size;
  const size_t sym_atomic_size;
  size_t getBufferSize() const;
};

}  // namespace ggnn
#endif  // GGNN_CONFIG_H