#pragma once

// Host-only. Included from .cu files only.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace lbm {

// Throws std::runtime_error with the CUDA error string, the failing
// expression, file and line if err != cudaSuccess.
inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " in '" +
                                 expr + "' at " + file + ":" + std::to_string(line));
    }
}

} // namespace lbm

#define LBM_CUDA_CHECK(expr) ::lbm::cuda_check((expr), #expr, __FILE__, __LINE__)
