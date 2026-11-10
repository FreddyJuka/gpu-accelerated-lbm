add_executable(lbm_tests
  test_sanity.cpp
  test_lattice_d2q9.cpp
  test_equilibrium.cpp
  test_moments.cpp
  test_collision_trt.cpp
  test_grid.cpp
  test_step_cpu.cpp
  test_bounce_back.cpp
  test_force.cpp
  test_poiseuille.cpp)

# The exact and bitwise tests assume IEEE arithmetic without contraction of
# a*b+c into FMA on the host (see support/fp_environment.hpp). Contraction is
# switched off explicitly for every test target, CXX and CUDA host passes alike,
# so that target flags such as -march=x86-64-v3 or -march=native (which enable
# FMA instructions) cannot change the results.
add_library(lbm_test_fp INTERFACE)
target_compile_options(lbm_test_fp INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
target_include_directories(lbm_test_fp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lbm_tests PRIVATE
  lbm::lbm
  lbm_warnings
  lbm_test_fp
  GTest::gtest_main)

gtest_discover_tests(lbm_tests)

if(LBM_ENABLE_CUDA)
  # Compile-only check of the device code path (never linked or launched).
  add_library(lbm_device_compile_check OBJECT device_compile_check.cu)
  target_link_libraries(lbm_device_compile_check PRIVATE lbm::lbm lbm_warnings lbm_test_fp)

  # GPU tests: exact CPU-vs-GPU comparison. Labelled "gpu" so that CI (no GPU)
  # can run `ctest -LE gpu`. Test discovery only lists tests and makes no CUDA
  # calls.
  add_executable(lbm_gpu_tests gpu/test_step_gpu.cu)
  target_link_libraries(lbm_gpu_tests PRIVATE
    lbm::lbm
    lbm_warnings
    lbm_test_fp
    GTest::gtest_main)
  # The exact CPU-vs-GPU comparison requires the same IEEE operations in the same
  # order, so FMA contraction is disabled here: --fmad=false on the device side
  # and -ffp-contract=off on the host side (the latter via lbm_test_fp). Production
  # code keeps nvcc's default FMA contraction.
  target_compile_options(lbm_gpu_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  gtest_discover_tests(lbm_gpu_tests PROPERTIES LABELS gpu)

  # GPU Poiseuille validation. Unlike lbm_gpu_tests it keeps nvcc's default FMA
  # contraction (production flags); the tolerance holds either way.
  add_executable(lbm_poiseuille_gpu_tests gpu/test_poiseuille_gpu.cu)
  target_link_libraries(lbm_poiseuille_gpu_tests PRIVATE
    lbm::lbm
    lbm_warnings
    lbm_test_fp
    GTest::gtest_main)
  gtest_discover_tests(lbm_poiseuille_gpu_tests PROPERTIES LABELS gpu)
endif()
