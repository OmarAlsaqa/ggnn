#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <torch/extension.h>

namespace ggnn {

/**
 * @brief A macro to check the result of a CUDA runtime API call.
 * If the call does not return cudaSuccess, it throws a C++ exception
 * that PyTorch converts into a Python RuntimeError.
 */
#define CHECK_CUDA(instruction)                                   \
  do {                                                            \
    cudaError_t res = instruction;                                \
    TORCH_CHECK(res == cudaSuccess, #instruction,                 \
                " failed with error: ", cudaGetErrorString(res)); \
  } while (0)
#define CHECK_CURAND(instruction)                           \
  do {                                                      \
    curandStatus_t res = instruction;                       \
    TORCH_CHECK(res == CURAND_STATUS_SUCCESS, #instruction, \
                " failed with cuRAND error code: ", res);   \
  } while (0)

}  // namespace ggnn