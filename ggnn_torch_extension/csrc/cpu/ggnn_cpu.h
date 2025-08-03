#ifndef GGNN_CPU_H
#define GGNN_CPU_H

#include <torch/extension.h>
#include <bit>    // For std::bit_ceil in C++20
#include <cmath>  // For std::pow, std::abs
#include <cstdint>
#include <filesystem>
#include <vector>
#include "../ggnn_config.h"

namespace ggnn {

// ============================================================================
// == PART 1: Core Definitions from the Original Project                     ==
// ============================================================================

// Moved to ggnn_config.h

// ============================================================================
// == PART 2: C++ API Function Declarations for the PyTorch Extension        ==
// ============================================================================

torch::Tensor build_graph_cpp(torch::Tensor base_tensor, int k_build, float tau_build,
                              int refinement_iterations, ggnn::DistanceMeasure measure);

std::tuple<torch::Tensor, torch::Tensor> query_graph_cpp(
    torch::Tensor query_tensor, torch::Tensor graph_tensor, torch::Tensor base_tensor,
    int k_build,  // Pass k_build to reconstruct config
    int k_query, float tau_query, int max_iterations, ggnn::DistanceMeasure measure);

std::tuple<torch::Tensor, torch::Tensor> bf_query_cpp(torch::Tensor base_tensor,
                                                      torch::Tensor query_tensor, int k,
                                                      ggnn::DistanceMeasure measure);

}  // namespace ggnn

#endif  // GGNN_CPU_H