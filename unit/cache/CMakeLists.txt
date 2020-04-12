add_executable(ssa_step_algorithm_test ssa_step_algorithm.test.cpp)
target_compile_definitions(ssa_step_algorithm_test PRIVATE "BOOST_TEST_DYN_LINK=1")

target_link_libraries(ssa_step_algorithm_test ansicfrontend cppfrontend clangcfrontend clangcppfrontend symex pointeranalysis langapi util_esbmc bigint solvers clibs ${Boost_LIBRARIES})
add_test(NAME SsaStepAlgorithm COMMAND ssa_step_algorithm_test)

add_executable(expr_variable_reordering_test expr_variable_reordering.test.cpp)
target_compile_definitions(expr_variable_reordering_test PRIVATE "BOOST_TEST_DYN_LINK=1")

target_link_libraries(expr_variable_reordering_test  smt_cache util_esbmc bigint ${Boost_LIBRARIES})
add_test(NAME ReorderingTest COMMAND expr_variable_reordering_test)

add_executable(green_normal_form_test green_normal_form.test.cpp)
target_compile_definitions(green_normal_form_test PRIVATE "BOOST_TEST_DYN_LINK=1")

target_link_libraries(green_normal_form_test smt_cache ansicfrontend cppfrontend clangcfrontend clangcppfrontend symex pointeranalysis langapi util_esbmc bigint solvers clibs ${Boost_LIBRARIES})
add_test(NAME GreenNormalFormTest COMMAND green_normal_form_test)

add_executable(green_storage_test green_storage.test.cpp)
target_compile_definitions(green_storage_test PRIVATE "BOOST_TEST_DYN_LINK=1")

target_link_libraries(green_storage_test smt_cache ansicfrontend cppfrontend clangcfrontend clangcppfrontend symex pointeranalysis langapi util_esbmc bigint solvers clibs ${Boost_LIBRARIES})
add_test(NAME GreenStorageTest COMMAND green_storage_test)

