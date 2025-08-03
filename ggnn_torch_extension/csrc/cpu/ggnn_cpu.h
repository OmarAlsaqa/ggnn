#ifndef GGNN_CPU_H
#define GGNN_CPU_H

#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <cmath> // For std::pow, std::abs
#include <bit>   // For std::bit_ceil in C++20

namespace ggnn {

// ============================================================================
// == PART 1: Core Definitions from the Original Project                     ==
// ============================================================================

enum class DistanceMeasure : int {
    Euclidean = 0,
    Cosine = 1
};

constexpr uint32_t MAX_NUM_LAYERS = 4;

template <typename T, T base>
constexpr T next_multiple(T v) noexcept {
    return v % base == 0 ? v : base * (v / base + 1);
};

inline size_t align8(size_t size) {
    return ((size + 7) / 8) * 8;
};

#if __cplusplus < 202002L
template <typename T>
constexpr T bit_ceil(T v) noexcept {
    v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16; v++;
    return v;
}
#else
using std::bit_ceil;
#endif

// --- GraphConfig and related structs (from original graph_config.h) ---
struct GraphParameters {
    uint32_t N{};
    uint32_t D{};
    uint32_t KBuild{};
};

struct GraphDerivedParameters : public GraphParameters {
    GraphDerivedParameters(const GraphParameters& params);
    uint32_t KF{KBuild / 2};
    static constexpr uint32_t L{MAX_NUM_LAYERS};
    uint32_t G{};
    uint32_t S{next_multiple<uint32_t, 32U>(KF + 1)};
    uint32_t S0{};
    uint32_t S0_off{};
    uint32_t SG{};
    uint32_t SG_off{};
};

struct GraphDimensions {
    GraphDimensions(const GraphDerivedParameters& params);
    uint32_t N_all{};
    uint32_t ST_all{};
    std::array<uint32_t, MAX_NUM_LAYERS> Bs{};
    std::array<uint32_t, MAX_NUM_LAYERS> Ns{};
    std::array<uint32_t, MAX_NUM_LAYERS> Ns_offsets{};
    std::array<uint32_t, MAX_NUM_LAYERS> STs_offsets{};
};

struct GraphConfig : public GraphDerivedParameters, public GraphDimensions {
    GraphConfig(const GraphParameters& params);
};

// --- PartSizes Structs (from original graph.h and graph_buffer.cuh) ---
// These are needed by the C++ host code to calculate tensor sizes.
namespace cuda {
    using KeyT = int32_t;
    using ValueT = float;
}

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


// ============================================================================
// == PART 2: C++ API Function Declarations for the PyTorch Extension        ==
// ============================================================================

torch::Tensor build_graph_cpp(
    torch::Tensor base_tensor,
    int k_build,
    float tau_build,
    int refinement_iterations,
    ggnn::DistanceMeasure measure
);

std::tuple<torch::Tensor, torch::Tensor> query_graph_cpp(
    torch::Tensor query_tensor,
    torch::Tensor graph_tensor,
    torch::Tensor base_tensor,
    int k_build, // Pass k_build to reconstruct config
    int k_query,
    float tau_query,
    int max_iterations,
    ggnn::DistanceMeasure measure
);

std::tuple<torch::Tensor, torch::Tensor> bf_query_cpp(
    torch::Tensor base_tensor,
    torch::Tensor query_tensor,
    int k,
    ggnn::DistanceMeasure measure
);

} // namespace ggnn

#endif // GGNN_CPU_H