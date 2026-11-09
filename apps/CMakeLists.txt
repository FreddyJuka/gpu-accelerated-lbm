add_executable(lbm_poiseuille poiseuille/main.cu)
target_link_libraries(lbm_poiseuille PRIVATE lbm::lbm lbm_warnings)
