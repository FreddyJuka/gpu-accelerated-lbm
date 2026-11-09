add_executable(lbm_tests
  test_sanity.cpp
  test_lattice_d2q9.cpp
  test_equilibrium.cpp
  test_moments.cpp
  test_collision_trt.cpp
  test_grid.cpp
  test_step_cpu.cpp
  test_bounce_back.cpp
  test_force.cpp)

target_link_libraries(lbm_tests PRIVATE
  lbm::lbm
  lbm_warnings
  GTest::gtest_main)

gtest_discover_tests(lbm_tests)

# Compile-only check of the device code path (never linked or launched).
add_library(lbm_device_compile_check OBJECT device_compile_check.cu)
target_link_libraries(lbm_device_compile_check PRIVATE lbm::lbm lbm_warnings)

# GPU tests: exact CPU-vs-GPU comparison. Labelled "gpu" so that CI (no GPU)
# can run `ctest -LE gpu`. Test discovery only lists tests and makes no CUDA
# calls.
add_executable(lbm_gpu_tests gpu/test_step_gpu.cu)
target_link_libraries(lbm_gpu_tests PRIVATE
  lbm::lbm
  lbm_warnings
  GTest::gtest_main)
# The exact CPU-vs-GPU comparison requires the same IEEE operations in the same
# order, so FMA contraction is disabled here. Production code keeps nvcc's
# default FMA contraction.
target_compile_options(lbm_gpu_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
gtest_discover_tests(lbm_gpu_tests PROPERTIES LABELS gpu)
