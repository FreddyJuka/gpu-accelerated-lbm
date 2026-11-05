#pragma once

// Host-only. Included from .cu files only.

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "lbm/cuda/check.hpp"

namespace lbm {

// Move-only RAII owner of a device array of `count` elements of T
// (cudaMalloc / cudaFree). A moved-from buffer has data() == nullptr and
// size() == 0. The destructor never throws.
template <typename T>
class DeviceBuffer {
public:
    // Throws std::invalid_argument if count <= 0, std::runtime_error if the
    // allocation fails.
    explicit DeviceBuffer(int count) {
        if (count <= 0)
            throw std::invalid_argument("DeviceBuffer: count must be > 0, got " +
                                        std::to_string(count));
        LBM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_),
                                  static_cast<std::size_t>(count) * sizeof(T)));
        size_ = count;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), size_(std::exchange(other.size_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = std::exchange(other.ptr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    ~DeviceBuffer() { release(); }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    int size() const { return size_; }

    // count must equal size(); otherwise std::invalid_argument.
    void copy_from_host(const T* src, int count) {
        require_size(count);
        LBM_CUDA_CHECK(cudaMemcpy(ptr_, src, bytes(), cudaMemcpyHostToDevice));
    }

    // count must equal size(); otherwise std::invalid_argument.
    void copy_to_host(T* dst, int count) const {
        require_size(count);
        LBM_CUDA_CHECK(cudaMemcpy(dst, ptr_, bytes(), cudaMemcpyDeviceToHost));
    }

private:
    std::size_t bytes() const { return static_cast<std::size_t>(size_) * sizeof(T); }

    void require_size(int count) const {
        if (count != size_)
            throw std::invalid_argument("DeviceBuffer: count " + std::to_string(count) +
                                        " does not match size " + std::to_string(size_));
    }

    // Errors from cudaFree cannot be reported from a destructor.
    void release() noexcept {
        if (ptr_ != nullptr) {
            try {
                LBM_CUDA_CHECK(cudaFree(ptr_));
            } catch (...) {
            }
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* ptr_ = nullptr;
    int size_ = 0;
};

} // namespace lbm
