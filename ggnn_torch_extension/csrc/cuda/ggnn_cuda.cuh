#pragma once

#include <torch/extension.h>
#include <array>
#include <vector>
#include <typeinfo> // For std::type_info
#include <cuda_runtime.h> // For cudaStream_t

// Forward-declare the host-side config struct.
// The full definition will be in csrc/cpu/ggnn_cpu.h.
namespace ggnn {
    struct GraphConfig;
    enum class DistanceMeasure;
}

struct curandGenerator_st;
typedef struct curandGenerator_st *curandGenerator_t;

namespace ggnn {
namespace cuda {

// Define fixed types for the PyTorch extension
using KeyT = int32_t;
using ValueT = float;

//==============================================================================
// 1. Non-owning "View" Structs (Directly maps to the GGNN algorithm)
//==============================================================================

// A non-owning view of the hierarchical GGNN graph structure.
struct GraphView {
    std::array<KeyT*, 4> graph;
    std::array<KeyT*, 4> translation;
    std::array<KeyT*, 4> selection;
    ValueT* nn1_stats;

    GraphView(const ggnn::GraphConfig& config, std::byte* graph_ptr);
};

// A non-owning view of the temporary buffer used during construction.
struct GraphBufferView {
    // Pointers into the temporary buffer tensor
    ValueT* nn1_dist_buffer;
    float* rng;
    std::byte* temp_storage_cub;
    size_t temp_storage_bytes_cub;
    KeyT* graph_buffer;
    KeyT* sym_buffer;
    uint32_t* sym_atomic;

    // Constructor calculates all pointers from a single base pointer.
    GraphBufferView(const ggnn::GraphConfig& config, std::byte* buffer_ptr);
};

//==============================================================================
// 2. Kernel Parameter Structs (Your excellent idea)
//==============================================================================
// We only need a public one for the final query. The construction kernel
// params can be internal to ggnn_cuda.cu as they are complex and varied.

struct QueryKernelParams {
    uint32_t D;
    DistanceMeasure measure;
    uint32_t KQuery;
    float tau_query;
    uint32_t max_iterations;
    
    // Pointers to Tensor data
    const void* d_base;
    const void* d_query;
    KeyT* d_query_results;
    ValueT* d_query_results_dists;

    // Pointers from the GraphView
    const KeyT* d_graph_layer0;
    const KeyT* d_starting_points;
    const float* d_nn1_stats;
    
    // Graph dimensions
    KeyT N_base;
    uint32_t KBuild;
    uint32_t num_starting_points;
    int64_t num_queries;
};

//==============================================================================
// 3. C++ Launcher Function Declarations (The Public CUDA API)
//==============================================================================

/**
 * @brief Orchestrates the entire graph construction process on the GPU.
 */
void launch_graph_construction(
    const ggnn::GraphConfig& config,
    GraphView& graph,
    GraphBufferView& buffer,
    const void* base_ptr,
    const std::type_info& base_type, // To select correct template instantiation
    float tau_build,
    ggnn::DistanceMeasure measure,
    uint32_t refinement_iterations,
    cudaStream_t stream
);

/**
 * @brief Launches the main query kernel.
 */
void launch_query(const QueryKernelParams& params, const std::type_info& base_type, cudaStream_t stream);

/**
 * @brief Launches the brute-force query kernel.
 */
void launch_bf_query(
    const void* base_ptr,
    const void* query_ptr,
    const std::type_info& base_type,
    KeyT* result_ids_ptr,
    ValueT* result_dists_ptr,
    int64_t N_base,
    int64_t N_query,
    int64_t D,
    int KQuery,
    ggnn::DistanceMeasure measure,
    cudaStream_t stream
);

} // namespace cuda
} // namespace ggnn