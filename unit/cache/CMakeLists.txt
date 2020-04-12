# Base libraries
add_library(cache_test_utils cache_test_utils.cpp)
target_compile_definitions(cache_test_utils PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(cache_test_utils cache_algorithms util_esbmc bigint ${Boost_LIBRARIES})

add_library(cache_fuzz_utils cache_fuzz_utils.cpp)
target_link_libraries(cache_fuzz_utils cache_test_utils)

# Expr Reordering Test
add_executable(expr_variable_reordering_test expr_variable_reordering.test.cpp)
target_link_libraries(expr_variable_reordering_test cache_test_utils)
add_test(NAME ReorderingTest COMMAND expr_variable_reordering_test)

# Cache Fuzz Utils Test
add_executable(cache_fuzz_utils_test cache_fuzz_utils.test.cpp)
target_link_libraries(cache_fuzz_utils_test cache_fuzz_utils)
add_test(NAME CacheFuzzUtilsTest COMMAND cache_fuzz_utils_test)





