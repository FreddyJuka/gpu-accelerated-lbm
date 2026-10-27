cmake_minimum_required(VERSION 3.24)

# Default GPU architecture: RTX 3050 Laptop (Ampere, sm_86).
# Must be set before project() so CMake does not pick the compiler default.
# Override with -DCMAKE_CUDA_ARCHITECTURES=<arch>.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(gpu_accelerated_lbm
  VERSION 0.1.0
  DESCRIPTION "GPU-accelerated Lattice Boltzmann CFD solver"
  LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release for single-config generators
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(LBM_BUILD_TESTS "Build unit tests" ON)

# ---------------------------------------------------------------------------
# Core library (header-only for now)
# ---------------------------------------------------------------------------
add_library(lbm INTERFACE)
add_library(lbm::lbm ALIAS lbm)
target_include_directories(lbm INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

# Warning flags, applied explicitly to project targets only
add_library(lbm_warnings INTERFACE)
target_compile_options(lbm_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

# ---------------------------------------------------------------------------
# Tests (GoogleTest via FetchContent, pinned)
# ---------------------------------------------------------------------------
if(LBM_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)
  add_subdirectory(tests)
endif()
