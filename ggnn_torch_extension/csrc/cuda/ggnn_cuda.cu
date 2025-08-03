#include "ggnn_cuda.cuh"
#include "utils/check.cuh"
#include "utils/simple_knn_cache.cuh"
#include "utils/simple_knn_sym_cache.cuh"

#include <curand.h>
#include <algorithm>  // For std::max, std::min
#include <cub/cub.cuh>
#include <memory>  // For std::unique_ptr

// --- Kernel Parameter Structs & __global__ Launchers ---
// We copy the original kernel structs and their __device__ operator() logic here.
// Each one is paired with a simple __global__ launcher function.

#include "kernels/bf_query_layer.cuh"
#include "kernels/merge_layer.cuh"
#include "kernels/query_layer.cuh"
#include "kernels/sym_buffer_merge_layer.cuh"
#include "kernels/sym_query_layer.cuh"
#include "kernels/top_merge_layer.cuh"
#include "kernels/wrs_select_layer.cuh"
namespace ggnn {
namespace cuda {

// =============================================================================
// == PART 1: VIEW STRUCT CONSTRUCTOR IMPLEMENTATIONS                         ==
// =============================================================================

GraphView::GraphView(const ggnn::GraphConfig& config, std::byte* graph_ptr)
{
  const ggnn::GraphPartSizes sizes(config);

  KeyT* neighborhood_data = reinterpret_cast<KeyT*>(graph_ptr);
  KeyT* selection_data = reinterpret_cast<KeyT*>(graph_ptr + sizes.graph_size);
  KeyT* translation_data =
      reinterpret_cast<KeyT*>(graph_ptr + sizes.graph_size + sizes.selection_translation_size);
  this->nn1_stats = reinterpret_cast<ValueT*>(graph_ptr + sizes.graph_size +
                                              2 * sizes.selection_translation_size);

  for (uint32_t layer = 0; layer < config.L; ++layer) {
    this->graph[layer] = neighborhood_data + config.Ns_offsets[layer] * config.KBuild;
    if (layer > 0) {
      this->selection[layer] = selection_data + config.STs_offsets[layer];
      this->translation[layer] = translation_data + config.STs_offsets[layer];
    }
    else {
      this->selection[layer] = nullptr;
      this->translation[layer] = nullptr;
    }
  }
}

GraphBufferView::GraphBufferView(const ggnn::GraphConfig& config, std::byte* buffer_ptr)
{
  const ggnn::GraphBufferPartSizes sizes(config);

  this->nn1_dist_buffer = reinterpret_cast<ValueT*>(buffer_ptr);
  this->graph_buffer = reinterpret_cast<KeyT*>(buffer_ptr + sizes.nn1_dist_buffer_size);
  this->rng = reinterpret_cast<float*>(buffer_ptr + sizes.nn1_dist_buffer_size);
  this->temp_storage_cub = buffer_ptr + sizes.nn1_dist_buffer_size;
  this->sym_buffer = reinterpret_cast<KeyT*>(buffer_ptr);
  this->sym_atomic = reinterpret_cast<uint32_t*>(buffer_ptr + sizes.sym_buffer_size);

  // Calculate CUB temporary storage size (as done in the original code)
  size_t temp_bytes = 0;
  cub::DeviceReduce::Sum(nullptr, temp_bytes, nn1_dist_buffer, nn1_dist_buffer, config.N);
  this->temp_storage_bytes_cub = temp_bytes;
}

namespace detail {

// =============================================================================
// == PART 2: INTERNAL KERNEL IMPLEMENTATIONS                                 ==
// == The contents of the original CUDA kernel files are ported here.         ==
// == These are implementation details and not part of the public CUDA API.   ==
// =============================================================================

// --- Utility Kernels ---
template <typename ValueT>
__global__ void divide_kernel(ValueT* res, ValueT* input, ValueT N)
{
  if (threadIdx.x == 0) {
    res[0] = input[0] / N;  // Mean
  }
  if (threadIdx.x == 1) {
    res[1] = input[1];  // Max
  }
}

// --- Internal C++ Launcher Helpers ---
// These helpers contain the logic for selecting template parameters and launching kernels.

// Helper to get a cuRAND generator
struct CurandGeneratorDeleter {
  void operator()(curandGenerator_t gen)
  {
    if (gen)
      curandDestroyGenerator(gen);
  }
};
using CurandGenerator = std::unique_ptr<curandGenerator_st, CurandGeneratorDeleter>;

static CurandGenerator createPRNG()
{
  curandGenerator_t gen;
  CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  return CurandGenerator(gen);
}

// Launcher for the brute-force/exhaustive search within a segment
template <typename BaseT>
void launch_top_merge_kernel(const ggnn::GraphConfig& config, GraphView& graph,
                             GraphBufferView& buffer, const BaseT* base_ptr,
                             ggnn::DistanceMeasure measure, cudaStream_t stream, uint32_t layer)
{
  // Logic from original GraphConstructionImpl::top()
  static constexpr uint32_t MIN_BLOCK_DIM_X = 128;
  const uint32_t dist_items_per_thread = config.D <= 1024 ? 4U : 8U;
  const uint32_t block_dim_x =
      std::max(MIN_BLOCK_DIM_X,
               ::ggnn::bit_ceil((config.D + dist_items_per_thread - 1) / dist_items_per_thread));

  auto launch = [&](auto block_dim, auto items_per_thread) {
    const uint32_t s_val = layer ? config.S : config.S0;
    const uint32_t s_offset_val = layer ? 0 : config.S0_off;
    detail::TopMergeKernel<KeyT, ValueT, BaseT, decltype(block_dim)::value,
                           decltype(items_per_thread)::value>
        kernel(config.D, measure, config.KBuild, base_ptr, graph.translation[layer],
               graph.graph[layer], buffer.nn1_dist_buffer, s_val, s_offset_val, layer);

    const size_t sm_size = config.KBuild * (sizeof(KeyT) + sizeof(ValueT));
    detail::top_merge_kernel_launcher<<<config.Ns[layer], block_dim_x, sm_size, stream>>>(kernel);
  };

  // Compile-time dispatch to the correct kernel template
  if (dist_items_per_thread == 4) {
    if (block_dim_x == 128)
      launch(std::integral_constant<uint32_t, 128>(), std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 256)
      launch(std::integral_constant<uint32_t, 256>(), std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 512)
      launch(std::integral_constant<uint32_t, 512>(), std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 1024)
      launch(std::integral_constant<uint32_t, 1024>(), std::integral_constant<uint32_t, 4>());
    else
      TORCH_CHECK(false,
                  "Unsupported block dimension for dist_items_per_thread=4 in top merge kernel.");
  }
  else if (dist_items_per_thread == 8) {
    if (block_dim_x == 128)
      launch(std::integral_constant<uint32_t, 128>(), std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 256)
      launch(std::integral_constant<uint32_t, 256>(), std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 512)
      launch(std::integral_constant<uint32_t, 512>(), std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 1024)
      launch(std::integral_constant<uint32_t, 1024>(), std::integral_constant<uint32_t, 8>());
    else
      TORCH_CHECK(false,
                  "Unsupported block dimension for dist_items_per_thread=8 in top merge kernel.");
  }
  else {
    TORCH_CHECK(false, "Unsupported dist_items_per_thread for top merge kernel.");
  }
}

// Launcher for the hierarchical graph search and merge
template <typename BaseT>
void launch_merge_kernel(const ggnn::GraphConfig& config, GraphView& graph, GraphBufferView& buffer,
                         const BaseT* base_ptr, ggnn::DistanceMeasure measure, float tau_build,
                         cudaStream_t stream, uint32_t layer_top, uint32_t layer_btm)
{
  // Logic from original GraphConstructionImpl::mergeLayer()
  static constexpr uint32_t MIN_BLOCK_DIM_X = 32;
  const uint32_t dist_items_per_thread = config.D <= 1024 ? 4U : 8U;
  const uint32_t block_dim_x =
      std::max(MIN_BLOCK_DIM_X,
               ::ggnn::bit_ceil((config.D + dist_items_per_thread - 1) / dist_items_per_thread));

  auto launch = [&](auto block_dim, auto items_per_thread) {
    detail::MergeKernel<KeyT, ValueT, BaseT, decltype(block_dim)::value,
                        decltype(items_per_thread)::value>
        kernel(config.D, measure, config.KBuild, base_ptr,
               graph.selection[1],      // d_selection
               graph.translation[1],    // d_translation
               graph.graph[0],          // d_graph
               buffer.graph_buffer,     // d_graph_buffer
               graph.nn1_stats,         // d_nn1_stats
               buffer.nn1_dist_buffer,  // d_nn1_dist_buffer
               config.S,                // S (was missing from call)
               layer_top,               // layer_top (order corrected)
               layer_btm,               // layer_btm
               config.G,                // G
               config.S0,               // S0
               config.S0_off,           // S0_offset
               config.Ns_offsets,       // Ns_offsets
               config.STs_offsets,      // STs_offsets
               tau_build                // tau_build
        );
    const size_t sm_size = kernel.CACHE_SIZE * sizeof(KeyT) + kernel.SORTED_SIZE * sizeof(ValueT);
    detail::merge_kernel_launcher<<<config.Ns[layer_btm], block_dim_x, sm_size, stream>>>(kernel);
  };

  if (dist_items_per_thread == 4) {
    if (block_dim_x == 32)
      launch(std::integral_constant<uint32_t, 32>(), std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 64)
      launch(std::integral_constant<uint32_t, 64>(), std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 128)
      launch(std::integral_constant<uint32_t, 128>(), std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 256)
      launch(std::integral_constant<uint32_t, 256>(), std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 512)
      launch(std::integral_constant<uint32_t, 512>(), std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 1024)
      launch(std::integral_constant<uint32_t, 1024>(), std::integral_constant<uint32_t, 4>());
    else
      TORCH_CHECK(false,
                  "Unsupported block dimension for dist_items_per_thread=4 in merge kernel.");
  }
  else if (dist_items_per_thread == 8) {
    if (block_dim_x == 32)
      launch(std::integral_constant<uint32_t, 32>(), std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 64)
      launch(std::integral_constant<uint32_t, 64>(), std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 128)
      launch(std::integral_constant<uint32_t, 128>(), std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 256)
      launch(std::integral_constant<uint32_t, 256>(), std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 512)
      launch(std::integral_constant<uint32_t, 512>(), std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 1024)
      launch(std::integral_constant<uint32_t, 1024>(), std::integral_constant<uint32_t, 8>());
    else
      TORCH_CHECK(false,
                  "Unsupported block dimension for dist_items_per_thread=8 in merge kernel.");
  }
  else {
    TORCH_CHECK(false, "Unsupported dist_items_per_thread for merge kernel.");
  }

  // Copy result back from temporary buffer to the main graph tensor
  const size_t graph_buffer_size =
      static_cast<size_t>(config.Ns[layer_btm]) * config.KBuild * sizeof(KeyT);
  CHECK_CUDA(cudaMemcpyAsync(graph.graph[layer_btm], buffer.graph_buffer, graph_buffer_size,
                             cudaMemcpyDeviceToDevice, stream));
}

// Launcher for the weighted random sampling selection
void launch_select_kernel(const ggnn::GraphConfig& config, GraphView& graph,
                          GraphBufferView& buffer, curandGenerator_t rand_generator,
                          cudaStream_t stream, uint32_t layer)
{
  // Logic from original GraphConstructionImpl::select()
  CHECK_CURAND(curandSetStream(rand_generator, stream));
  CHECK_CURAND(curandGenerateUniform(rand_generator, buffer.rng, config.Ns[layer]));

  detail::WRSSelectionKernel<KeyT, ValueT> kernel(
      graph.selection[layer + 1], graph.translation[layer + 1], graph.translation[layer],
      buffer.nn1_dist_buffer, buffer.rng, config.S, layer ? config.S : config.S0,
      layer ? 0 : config.S0_off, config.G, config.SG, config.SG_off, layer);

  detail::select_kernel_launcher<<<config.Bs[layer], kernel.BLOCK_DIM_X, 0, stream>>>(kernel);
}

// Launcher for the symmetric link diversification step
template <typename BaseT>
void launch_sym_step(const ggnn::GraphConfig& config, GraphView& graph, GraphBufferView& buffer,
                     const BaseT* base_ptr, ggnn::DistanceMeasure measure, float tau_build,
                     cudaStream_t stream, uint32_t layer)
{
  // Logic from original GraphConstructionImpl::sym()

  // 1. Clear temporary symmetric-link buffers
  CHECK_CUDA(cudaMemsetAsync(buffer.sym_buffer, -1,
                             static_cast<size_t>(config.Ns[layer]) * config.KF * sizeof(KeyT),
                             stream));
  CHECK_CUDA(cudaMemsetAsync(buffer.sym_atomic, 0, config.Ns[layer] * sizeof(uint32_t), stream));

  // 2. Launch SymQueryKernel to find missing links
  static constexpr uint32_t MIN_BLOCK_DIM_X = 64;
  const uint32_t dist_items_per_thread = config.D <= 1024 ? 4U : 8U;
  const uint32_t block_dim_x =
      std::max(MIN_BLOCK_DIM_X,
               ::ggnn::bit_ceil((config.D + dist_items_per_thread - 1) / dist_items_per_thread));

  auto launch_sym_query = [&](auto block_dim, auto items_per_thread) {
    detail::SymQueryKernel<KeyT, ValueT, BaseT, decltype(block_dim)::value,
                           decltype(items_per_thread)::value>
        kernel(config.D, measure, config.KBuild, base_ptr, graph.graph[layer],
               graph.translation[layer], graph.nn1_stats, tau_build, buffer.sym_buffer,
               buffer.sym_atomic);
    const size_t sm_size = kernel.CACHE_SIZE * sizeof(KeyT) + kernel.sorted_size * sizeof(ValueT);
    detail::sym_query_kernel_launcher<<<config.Ns[layer], block_dim_x, sm_size, stream>>>(kernel);
  };

  if (dist_items_per_thread == 4) {
    if (block_dim_x == 64)
      launch_sym_query(std::integral_constant<uint32_t, 64>(),
                       std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 128)
      launch_sym_query(std::integral_constant<uint32_t, 128>(),
                       std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 256)
      launch_sym_query(std::integral_constant<uint32_t, 256>(),
                       std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 512)
      launch_sym_query(std::integral_constant<uint32_t, 512>(),
                       std::integral_constant<uint32_t, 4>());
    else if (block_dim_x == 1024)
      launch_sym_query(std::integral_constant<uint32_t, 1024>(),
                       std::integral_constant<uint32_t, 4>());
    else
      TORCH_CHECK(false,
                  "Unsupported block dimension for dist_items_per_thread=4 in sym query kernel.");
  }
  else if (dist_items_per_thread == 8) {
    if (block_dim_x == 64)
      launch_sym_query(std::integral_constant<uint32_t, 64>(),
                       std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 128)
      launch_sym_query(std::integral_constant<uint32_t, 128>(),
                       std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 256)
      launch_sym_query(std::integral_constant<uint32_t, 256>(),
                       std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 512)
      launch_sym_query(std::integral_constant<uint32_t, 512>(),
                       std::integral_constant<uint32_t, 8>());
    else if (block_dim_x == 1024)
      launch_sym_query(std::integral_constant<uint32_t, 1024>(),
                       std::integral_constant<uint32_t, 8>());
    else
      TORCH_CHECK(false,
                  "Unsupported block dimension for dist_items_per_thread=8 in sym query kernel.");
  }
  else {
    TORCH_CHECK(false, "Unsupported dist_items_per_thread for sym query kernel.");
  }

  // 3. Launch SymBufferMergeKernel to write the new links into the graph
  detail::SymBufferMergeKernel<KeyT, ValueT> merge_kernel(config.KBuild, buffer.sym_buffer,
                                                          buffer.sym_atomic, graph.graph[layer]);
  const uint32_t N = config.Ns[layer];
  const dim3 block(merge_kernel.KF, merge_kernel.POINTS_PER_BLOCK);
  const size_t sm_size = sizeof(KeyT) * merge_kernel.POINTS_PER_BLOCK * merge_kernel.KF * 2 +
                         sizeof(bool) * merge_kernel.POINTS_PER_BLOCK;
  detail::sym_buffer_merge_kernel_launcher<<<(N - 1) / merge_kernel.POINTS_PER_BLOCK + 1, block,
                                             sm_size, stream>>>(merge_kernel, N);
}

void compute_nn1_stats(const ggnn::GraphConfig& config, GraphView& graph, GraphBufferView& buffer,
                       cudaStream_t stream)
{
  cub::DeviceReduce::Sum(buffer.temp_storage_cub, buffer.temp_storage_bytes_cub,
                         buffer.nn1_dist_buffer, graph.nn1_stats, config.N, stream);
  cub::DeviceReduce::Max(buffer.temp_storage_cub, buffer.temp_storage_bytes_cub,
                         buffer.nn1_dist_buffer, graph.nn1_stats + 1, config.N, stream);
  detail::divide_kernel<ValueT>
      <<<1, 2, 0, stream>>>(graph.nn1_stats, graph.nn1_stats, (ValueT)config.N);
}

// Launcher for the main query
template <typename BaseT>
void launch_query_kernel_typed(const ggnn::cuda::QueryKernelParams& params, cudaStream_t stream)
{
  // Logic from original QueryKernelsImpl::query
  static constexpr uint32_t WARP_SIZE = 32;
  static constexpr uint32_t DIST_ITEMS_PER_THREAD = 4;
  const uint32_t dimension_block_dim_x =
      ::ggnn::bit_ceil((params.D + DIST_ITEMS_PER_THREAD - 1) / DIST_ITEMS_PER_THREAD);
  // Calculate cache parameters
  const uint32_t required_sorted_size =
      ::ggnn::next_multiple<uint32_t, 32>(params.KQuery + 1 + 16 /*MIN_PRIOQ_SIZE*/);
  const uint32_t cache_size = std::max({256u, required_sorted_size + 32 /*MIN_VISITED_SIZE*/,
                                        ::ggnn::bit_ceil(params.max_iterations)});
  const uint32_t cache_size_block_dim_x =
      ::ggnn::bit_ceil((cache_size + 16 /*CACHE_ITEMS_PER_THREAD*/ - 1) / 16);
  const uint32_t block_dim_x = std::max({WARP_SIZE, dimension_block_dim_x, cache_size_block_dim_x});
  const uint32_t sorted_size = std::max(cache_size < 512U ? 64U : 32U, required_sorted_size);
  TORCH_CHECK(block_dim_x <= 1024, "Calculated block dimension exceeds CUDA limits.");
  auto launch = [&](auto block_dim) {
    detail::QueryKernel<KeyT, ValueT, BaseT, decltype(block_dim)::value, true, false> kernel(
        params.D, params.measure, params.KQuery, sorted_size, cache_size, params.tau_query,
        params.max_iterations, params.N_base, params.KBuild, params.num_starting_points,
        static_cast<const BaseT*>(params.d_base), static_cast<const BaseT*>(params.d_query),
        params.d_graph_layer0, params.d_starting_points, params.d_nn1_stats, params.d_query_results,
        params.d_query_results_dists,
        nullptr,  // d_dist_stats (not used in this version)
        1,        // shards_per_gpu (simplified)
        0         // on_gpu_shard_id (simplified)
    );
    const size_t sm_size = kernel.cache_size * sizeof(KeyT) + kernel.sorted_size * sizeof(ValueT);
    TORCH_CHECK(sm_size < 48 * 1024, "Shared memory usage exceeds typical limits.");

    detail::query_kernel_launcher<<<params.num_queries, block_dim_x, sm_size, stream>>>(kernel);
  };

  if (block_dim_x <= 32)
    launch(std::integral_constant<uint32_t, 32>());
  else if (block_dim_x <= 64)
    launch(std::integral_constant<uint32_t, 64>());
  else if (block_dim_x <= 128)
    launch(std::integral_constant<uint32_t, 128>());
  else if (block_dim_x <= 256)
    launch(std::integral_constant<uint32_t, 256>());
  else if (block_dim_x <= 512)
    launch(std::integral_constant<uint32_t, 512>());
  else if (block_dim_x <= 1024)
    launch(std::integral_constant<uint32_t, 1024>());
  else {
    TORCH_CHECK(false, "Unsupported block dimension for query kernel: ", block_dim_x);
  }
}

// Launcher for the brute force query
template <typename BaseT>
void launch_bf_query_kernel_typed(const BaseT* d_base, const BaseT* d_query, KeyT* d_out_ids,
                                  ValueT* d_out_dists, int64_t num_base, int64_t num_queries,
                                  int64_t dim, int k, ggnn::DistanceMeasure measure,
                                  cudaStream_t stream)
{
  // Logic from original QueryKernelsImpl::bruteForceQuery
  static constexpr uint32_t WARP_SIZE = 32;
  const uint32_t dist_items_per_thread = 4;
  const uint32_t dimension_block_dim_x =
      ::ggnn::bit_ceil((uint32_t)(dim + dist_items_per_thread - 1) / dist_items_per_thread);
  const uint32_t block_dim_x = std::max(WARP_SIZE, dimension_block_dim_x);
  TORCH_CHECK(block_dim_x <= 1024, "Calculated block dimension exceeds CUDA limits for bf_query.");

  // Construct QueryKernelParams from the function arguments
  auto launch = [&](auto block_dim_const) {
    // Construct the kernel directly from the function arguments.
    // Do NOT use QueryKernelParams here.
    BruteForceQueryKernel<KeyT, ValueT, BaseT, decltype(block_dim_const)::value, true> kernel(
        (uint32_t)dim, measure, (uint32_t)k, (KeyT)num_base, d_base, d_query, d_out_ids,
        d_out_dists);

    const size_t sm_size = k * (sizeof(KeyT) + sizeof(ValueT));
    bf_query_kernel_launcher<<<num_queries, block_dim_x, sm_size, stream>>>(kernel);
  };
  if (block_dim_x <= 32)
    launch(std::integral_constant<uint32_t, 32>());
  else if (block_dim_x <= 64)
    launch(std::integral_constant<uint32_t, 64>());
  else if (block_dim_x <= 128)
    launch(std::integral_constant<uint32_t, 128>());
  else if (block_dim_x <= 256)
    launch(std::integral_constant<uint32_t, 256>());
  else if (block_dim_x <= 512)
    launch(std::integral_constant<uint32_t, 512>());
  else if (block_dim_x <= 1024)
    launch(std::integral_constant<uint32_t, 1024>());
  else {
    TORCH_CHECK(false, "Unsupported block dimension for bf query kernel.");
  }
}

}  // namespace detail

// =============================================================================
// == PART 3: PUBLIC C++ LAUNCHER IMPLEMENTATIONS                             ==
// =============================================================================

// --- Launcher Implementations ---

// This is the full implementation for the public launcher function in csrc/ggnn_cuda.cu

void launch_graph_construction(const ggnn::GraphConfig& config, GraphView& graph,
                               GraphBufferView& buffer, const void* base_ptr_void,
                               const std::type_info& base_type, float tau_build,
                               ggnn::DistanceMeasure measure, uint32_t refinement_iterations,
                               cudaStream_t stream)
{
  // This function orchestrates the entire build process by calling the internal
  // detail:: launchers in the correct hierarchical order.

  // Get a cuRAND generator for the selection kernel
  auto rand_generator = detail::createPRNG();

  // A helper lambda to avoid duplicating the main build loops for float and uint8_t
  auto build_for_type = [&](auto base_ptr) {
    using BaseT = std::remove_const_t<std::remove_pointer_t<decltype(base_ptr)>>;

    // --- Main Hierarchical Build Loop ---
    for (uint32_t layer_top = 0; layer_top < config.L; ++layer_top) {
      for (uint32_t layer_btm = layer_top; layer_btm != -1U; --layer_btm) {
        // 1. MERGE STEP: Find nearest neighbors
        if (layer_top == layer_btm) {
          // This is a "top" layer merge: brute-force search within a segment.
          detail::launch_top_merge_kernel<BaseT>(config, graph, buffer, base_ptr, measure, stream,
                                                 layer_btm);
        }
        else {
          // This is a hierarchical merge: use the graph from a higher layer to guide the search.
          detail::launch_merge_kernel<BaseT>(config, graph, buffer, base_ptr, measure, tau_build,
                                             stream, layer_top, layer_btm);
        }

        // After merging the base layer, calculate statistics for the stopping criterion.
        if (layer_btm == 0) {
          detail::compute_nn1_stats(config, graph, buffer, stream);
        }

        // 2. SELECT STEP: Choose points for the next level of the hierarchy
        if (layer_top < (config.L - 1) && layer_top == layer_btm) {
          detail::launch_select_kernel(config, graph, buffer, rand_generator.get(), stream,
                                       layer_top);
        }

        // 3. SYM STEP: Add symmetric links to improve graph connectivity
        detail::launch_sym_step<BaseT>(config, graph, buffer, base_ptr, measure, tau_build, stream,
                                       layer_btm);
      }
    }

    // --- Refinement Loop ---
    for (uint32_t i = 0; i < refinement_iterations; ++i) {
      for (uint32_t layer = config.L - 2; layer != -1U; --layer) {
        // Re-run merge and sym steps to refine the graph connections
        detail::launch_merge_kernel<BaseT>(config, graph, buffer, base_ptr, measure, tau_build,
                                           stream, config.L - 1, layer);
        detail::launch_sym_step<BaseT>(config, graph, buffer, base_ptr, measure, tau_build, stream,
                                       layer);
      }
    }
  };

  // Dispatch to the correctly typed lambda based on the input tensor's type
  if (base_type == typeid(float)) {
    build_for_type(static_cast<const float*>(base_ptr_void));
  }
  else if (base_type == typeid(uint8_t)) {
    build_for_type(static_cast<const uint8_t*>(base_ptr_void));
  }
  else {
    TORCH_CHECK(false, "Unsupported base data type for GGNN construction");
  }
}

void launch_query(const QueryKernelParams& params, const std::type_info& base_type,
                  cudaStream_t stream)
{
  if (base_type == typeid(float)) {
    detail::launch_query_kernel_typed<float>(params, stream);
  }
  else if (base_type == typeid(uint8_t)) {
    detail::launch_query_kernel_typed<uint8_t>(params, stream);
  }
  else {
    TORCH_CHECK(false, "Unsupported base data type for GGNN query");
  }
}

void launch_bf_query(const void* base_ptr_void, const void* query_ptr_void,
                     const std::type_info& base_type, KeyT* result_ids_ptr,
                     ValueT* result_dists_ptr, int64_t N_base, int64_t N_query, int64_t D,
                     int KQuery, ggnn::DistanceMeasure measure, cudaStream_t stream)
{
  if (base_type == typeid(float)) {
    detail::launch_bf_query_kernel_typed<float>(
        static_cast<const float*>(base_ptr_void), static_cast<const float*>(query_ptr_void),
        result_ids_ptr, result_dists_ptr, N_base, N_query, D, KQuery, measure, stream);
  }
  else if (base_type == typeid(uint8_t)) {
    detail::launch_bf_query_kernel_typed<uint8_t>(
        static_cast<const uint8_t*>(base_ptr_void), static_cast<const uint8_t*>(query_ptr_void),
        result_ids_ptr, result_dists_ptr, N_base, N_query, D, KQuery, measure, stream);
  }
  else {
    TORCH_CHECK(false, "Unsupported base data type for GGNN bf query");
  }
}

}  // namespace cuda
}  // namespace ggnn