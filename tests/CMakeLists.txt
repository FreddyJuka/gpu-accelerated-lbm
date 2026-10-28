add_executable(lbm_tests
  test_sanity.cpp
  test_lattice_d2q9.cpp
  test_equilibrium.cpp
  test_moments.cpp
  test_collision_trt.cpp)

target_link_libraries(lbm_tests PRIVATE
  lbm::lbm
  lbm_warnings
  GTest::gtest_main)

gtest_discover_tests(lbm_tests)

# Compile-only check of the device code path (never linked or launched).
add_library(lbm_device_compile_check OBJECT device_compile_check.cu)
target_link_libraries(lbm_device_compile_check PRIVATE lbm::lbm lbm_warnings)
