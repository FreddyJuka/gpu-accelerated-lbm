cmake_minimum_required(VERSION 3.24)

project(gpu_accelerated_lbm
  VERSION 0.1.0
  DESCRIPTION "GPU-accelerated Lattice Boltzmann CFD solver"
  LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(LBM_ENABLE_CUDA "Build CUDA code" ON)
option(LBM_BUILD_TESTS "Build unit tests" ON)
option(LBM_BUILD_APPS "Build the example and measurement applications" ON)
option(LBM_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# ---------------------------------------------------------------------------
# CUDA (optional). Never falls back silently: if CUDA is requested but no
# compiler is found, configuration fails.
# ---------------------------------------------------------------------------
if(LBM_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "LBM_ENABLE_CUDA is ON but no CUDA compiler was found. Install the CUDA "
      "toolkit (nvcc) or configure with -DLBM_ENABLE_CUDA=OFF to build the "
      "CPU-only parts.")
  endif()

  # Default GPU architecture: RTX 3050 Laptop (Ampere, sm_86).
  # Must be set before enable_language(CUDA) so CMake does not pick the
  # compiler default. Override with -DCMAKE_CUDA_ARCHITECTURES=<arch>.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  enable_language(CUDA)
endif()

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

# Only targets linking lbm_warnings are affected; GoogleTest is not.
if(LBM_WARNINGS_AS_ERRORS)
  target_compile_options(lbm_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    # nvcc's own warnings, then the host compiler pass
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror all-warnings>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
endif()

# ---------------------------------------------------------------------------
# Tests (GoogleTest via FetchContent, pinned)
# ---------------------------------------------------------------------------
if(LBM_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
    URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
if(LBM_BUILD_APPS)
  if(LBM_ENABLE_CUDA)
    add_subdirectory(apps)
  else()
    message(STATUS "LBM_ENABLE_CUDA is OFF: skipping apps (they need nvcc)")
  endif()
endif()
