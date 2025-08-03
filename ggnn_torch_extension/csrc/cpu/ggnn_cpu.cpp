#include "ggnn_cpu.h"
#include "../cuda/ggnn_cuda.cuh" // The interface to our CUDA code
#include <ATen/cuda/CUDAContext.h> // For getting the current CUDA stream

#include <cmath> // For std::pow, std::abs
#include <glog/logging.h> // Optional: for detailed logging

namespace ggnn {

// ============================================================================
// == PART 1: GraphConfig Implementation (Copied from original project)      ==
// ============================================================================

// --- Helper function from original graph_config.cpp ---
constexpr uint32_t powInt(const uint32_t base, const uint32_t power) {
    if (!power) return 1;
    if (power == 1) return base;
    return base * powInt(base, power - 1);
}

// --- Constructor Implementations ---
GraphDerivedParameters::GraphDerivedParameters(const GraphParameters& params) : GraphParameters{params} {
    const float growth = std::pow(static_cast<float>(params.N) / static_cast<float>(S), 1.f / (L - 1));
    const uint32_t Gf = static_cast<uint32_t>(growth);
    const uint32_t Gc = Gf + 1;
    const float S0f = static_cast<float>(N) / (std::pow(static_cast<float>(Gf), (L - 1.0f)));
    const float S0c = static_cast<float>(N) / (std::pow(static_cast<float>(Gc), (L - 1.0f)));
    const bool is_floor = (static_cast<uint32_t>(S0c) < KBuild) || (std::abs(S0f - static_cast<float>(S)) < std::abs(S0c - static_cast<float>(S)));
    G = (is_floor) ? Gf : Gc;
    S0 = (is_floor) ? static_cast<uint32_t>(S0f) : static_cast<uint32_t>(S0c);
    S0_off = N - powInt(G, L - 1) * S0;
    SG = S / G;
    SG_off = S - SG * G;
}

GraphDimensions::GraphDimensions(const GraphDerivedParameters& params) {
    for (uint32_t l = params.L - 1, B = 1; l != -1U; --l, B *= params.G) {
        Bs[l] = B;
        Ns[l] = B * params.S;
    }
    Ns[0] = params.N;
    Ns_offsets[0] = 0;
    STs_offsets[0] = 0;
    STs_offsets[1] = 0;
    Ns_offsets[1] = Ns[0];
    for (uint32_t l = 2; l < params.L; ++l) {
        Ns_offsets[l] = Ns_offsets[l - 1] + Ns[l - 1];
        STs_offsets[l] = STs_offsets[l - 1] + Ns[l - 1];
    }
    N_all = Ns_offsets[params.L - 1] + Ns[params.L - 1];
    ST_all = STs_offsets[params.L - 1] + Ns[params.L - 1];
}

GraphConfig::GraphConfig(const GraphParameters& params)
    : GraphDerivedParameters{params}, GraphDimensions{*static_cast<GraphDerivedParameters*>(this)} {}

// --- PartSizes Constructor Implementations ---
GraphPartSizes::GraphPartSizes(const GraphConfig& config)
    : graph_size(align8(static_cast<size_t>(config.N_all) * config.KBuild * sizeof(cuda::KeyT))),
      selection_translation_size(align8(static_cast<size_t>(config.ST_all) * sizeof(cuda::KeyT))),
      nn1_stats_size(align8(2UL * sizeof(cuda::ValueT))) {}

size_t GraphPartSizes::getGraphSize() const {
    return graph_size + 2 * selection_translation_size + nn1_stats_size;
}

GraphBufferPartSizes::GraphBufferPartSizes(const GraphConfig& config)
    : graph_buffer_size(align8(static_cast<size_t>(config.N) * config.KBuild * sizeof(cuda::KeyT))),
      nn1_dist_buffer_size(align8(static_cast<size_t>(config.N) * sizeof(cuda::ValueT))),
      rng_size(align8(static_cast<size_t>(config.N) * sizeof(float))),
      sym_buffer_size(align8(static_cast<size_t>(config.N) * config.KF * sizeof(cuda::KeyT))),
      sym_atomic_size(align8(static_cast<size_t>(config.N) * sizeof(uint32_t))) {}

size_t GraphBufferPartSizes::getBufferSize() const {
    const size_t merge_size = nn1_dist_buffer_size + graph_buffer_size;
    const size_t select_size = nn1_dist_buffer_size + rng_size;
    const size_t sym_size = sym_buffer_size + sym_atomic_size;
    return std::max({merge_size, select_size, sym_size});
}


// ============================================================================
// == PART 2: C++ API Function Implementations                               ==
// ============================================================================

torch::Tensor build_graph_cpp(
    torch::Tensor base_tensor,
    int k_build,
    float tau_build,
    int refinement_iterations,
    ggnn::DistanceMeasure measure)
{
    // 1. --- Input Validation ---
    TORCH_CHECK(base_tensor.is_cuda(), "Input 'base_tensor' must be a CUDA tensor.");
    TORCH_CHECK(base_tensor.dim() == 2, "Input 'base_tensor' must be 2D.");
    TORCH_CHECK(base_tensor.is_contiguous(), "Input 'base_tensor' must be contiguous.");
    const auto base_type = base_tensor.scalar_type();
    TORCH_CHECK(base_type == torch::kFloat32 || base_type == torch::kUInt8, "base_tensor must be of type float32 or uint8.");

    // 2. --- Configuration ---
    ggnn::GraphParameters params{
        .N = (uint32_t)base_tensor.size(0),
        .D = (uint32_t)base_tensor.size(1),
        .KBuild = (uint32_t)k_build
    };
    ggnn::GraphConfig config(params);

    // 3. --- Tensor Allocation (using PyTorch) ---
    const ggnn::GraphPartSizes graph_sizes(config);
    const size_t graph_total_bytes = graph_sizes.getGraphSize();

    const ggnn::GraphBufferPartSizes buffer_sizes(config);
    const size_t buffer_total_bytes = buffer_sizes.getBufferSize();

    auto byte_options = torch::TensorOptions().device(base_tensor.device()).dtype(torch::kUInt8);
    
    // Allocate the single, monolithic tensor for the final graph.
    torch::Tensor graph_tensor = torch::empty({(int64_t)graph_total_bytes}, byte_options);

    // Allocate the temporary workspace buffer. It will be freed automatically when it goes out of scope.
    torch::Tensor buffer_tensor = torch::empty({(int64_t)buffer_total_bytes}, byte_options);

    // 4. --- Create Non-owning Views and Get Pointers ---
    auto graph_ptr = reinterpret_cast<std::byte*>(graph_tensor.data_ptr());
    auto buffer_ptr = reinterpret_cast<std::byte*>(buffer_tensor.data_ptr());
    
    ggnn::cuda::GraphView graph_view(config, graph_ptr);
    ggnn::cuda::GraphBufferView buffer_view(config, buffer_ptr);
    
    const void* base_ptr = base_tensor.data_ptr();
    const auto& type_info = base_tensor.dtype().toScalarType() == torch::kFloat ? typeid(float) : typeid(uint8_t);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 5. --- Launch the CUDA Backend ---
    ggnn::cuda::launch_graph_construction(
        config,
        graph_view,
        buffer_view,
        base_ptr,
        type_info,
        tau_build,
        measure,
        (uint32_t)refinement_iterations,
        stream
    );

    return graph_tensor;
}


std::tuple<torch::Tensor, torch::Tensor> query_graph_cpp(
    torch::Tensor query_tensor,
    torch::Tensor graph_tensor,
    torch::Tensor base_tensor,
    int k_build, 
    int k_query,
    float tau_query,
    int max_iterations,
    ggnn::DistanceMeasure measure)
{
    // 1. --- Input Validation ---
    TORCH_CHECK(query_tensor.is_cuda() && graph_tensor.is_cuda() && base_tensor.is_cuda(), "All input tensors must be CUDA tensors.");
    TORCH_CHECK(query_tensor.dim() == 2, "Query tensor must be 2D.");
    TORCH_CHECK(query_tensor.device() == base_tensor.device() && query_tensor.device() == graph_tensor.device(), "All tensors must be on the same CUDA device.");
    TORCH_CHECK(query_tensor.dtype() == base_tensor.dtype(), "Query and base tensors must have the same dtype.");

    // 2. --- Configuration ---
    ggnn::GraphParameters params{
        .N = (uint32_t)base_tensor.size(0),
        .D = (uint32_t)base_tensor.size(1),
        .KBuild = (uint32_t)k_build
    };
    ggnn::GraphConfig config(params);

    // 3. --- Allocate Output Tensors ---
    const int64_t num_queries = query_tensor.size(0);
    auto id_options = torch::TensorOptions().device(query_tensor.device()).dtype(torch::kInt32);
    auto dist_options = torch::TensorOptions().device(query_tensor.device()).dtype(torch::kFloat32);

    torch::Tensor result_ids = torch::empty({num_queries, (long)k_query}, id_options);
    torch::Tensor result_dists = torch::empty({num_queries, (long)k_query}, dist_options);

    // 4. --- Create Views and Get Pointers ---
    auto graph_ptr = reinterpret_cast<std::byte*>(graph_tensor.data_ptr());
    ggnn::cuda::GraphView graph_view(config, graph_ptr);
    
    const auto& type_info = query_tensor.dtype().toScalarType() == torch::kFloat ? typeid(float) : typeid(uint8_t);
    
    // 5. --- Populate Kernel Params and Launch ---
    ggnn::cuda::QueryKernelParams kernel_params {
        .D = (uint32_t)query_tensor.size(1),
        .measure = measure,
        .KQuery = (uint32_t)k_query,
        .tau_query = tau_query,
        .max_iterations = (uint32_t)max_iterations,
        .d_base = base_tensor.data_ptr(),
        .d_query = query_tensor.data_ptr(),
        .d_query_results = result_ids.data_ptr<ggnn::cuda::KeyT>(),
        .d_query_results_dists = result_dists.data_ptr<ggnn::cuda::ValueT>(),
        .d_graph_layer0 = graph_view.graph[0],
        .d_starting_points = graph_view.translation[config.L - 1],
        .d_nn1_stats = graph_view.nn1_stats,
        .N_base = (ggnn::cuda::KeyT)base_tensor.size(0),
        .KBuild = config.KBuild,
        .num_starting_points = config.S
    };

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ggnn::cuda::launch_query(kernel_params, type_info, stream);

    return {result_ids, result_dists};
}

std::tuple<torch::Tensor, torch::Tensor> bf_query_cpp(
    torch::Tensor base_tensor,
    torch::Tensor query_tensor,
    int k,
    ggnn::DistanceMeasure measure)
{
    TORCH_CHECK(base_tensor.is_cuda() && query_tensor.is_cuda(), "Tensors must be CUDA tensors.");
    // TODO: review these checks. 
    TORCH_CHECK(base_tensor.dtype() == query_tensor.dtype(), "Base and query tensors must have the same dtype.");
    TORCH_CHECK(base_tensor.dim() == 2 && query_tensor.dim() == 2, "Both tensors must be 2D.");
    TORCH_CHECK(base_tensor.size(1) == query_tensor.size(1), "Base and query tensors must have the same number of dimensions (D).");
    TORCH_CHECK(k > 0, "k must be greater than 0.");
    TORCH_CHECK(base_tensor.is_contiguous() && query_tensor.is_contiguous(), "Both tensors must be contiguous.");

    const auto& type_info = query_tensor.dtype().toScalarType() == torch::kFloat ? typeid(float) : typeid(uint8_t);

    const int64_t num_queries = query_tensor.size(0);
    auto id_options = torch::TensorOptions().device(query_tensor.device()).dtype(torch::kInt32);
    auto dist_options = torch::TensorOptions().device(query_tensor.device()).dtype(torch::kFloat32);
    
    torch::Tensor result_ids = torch::empty({num_queries, (long)k}, id_options);
    torch::Tensor result_dists = torch::empty({num_queries, (long)k}, dist_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    ggnn::cuda::launch_bf_query(
        base_tensor.data_ptr(),
        query_tensor.data_ptr(),
        type_info,
        result_ids.data_ptr<ggnn::cuda::KeyT>(),
        result_dists.data_ptr<ggnn::cuda::ValueT>(),
        base_tensor.size(0),
        query_tensor.size(0),
        query_tensor.size(1),
        k,
        measure,
        stream
    );

    return {result_ids, result_dists};
}

} // namespace ggnn