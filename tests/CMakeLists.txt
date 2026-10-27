add_executable(lbm_tests
  test_sanity.cpp
  test_lattice_d2q9.cpp
  test_equilibrium.cpp)

target_link_libraries(lbm_tests PRIVATE
  lbm::lbm
  lbm_warnings
  GTest::gtest_main)

gtest_discover_tests(lbm_tests)
