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

# Expr Green Normal Form Test
add_executable(expr_green_normal_form_test expr_green_normal_form.test.cpp)
target_link_libraries(expr_green_normal_form_test cache_test_utils)
add_test(NAME GreenNormalFormTest COMMAND expr_green_normal_form_test)

# Cache Fuzz Utils Test
add_executable(cache_fuzz_utils_test cache_fuzz_utils.test.cpp)
target_link_libraries(cache_fuzz_utils_test cache_fuzz_utils)
add_test(NAME CacheFuzzUtilsTest COMMAND cache_fuzz_utils_test)


if(ENABLE_FUZZER)
    # Expr Reordering Fuzz
    add_executable(expr_variable_reordering_fuzz expr_variable_reordering.fuzz.cpp)
    add_test(NAME ReorderingFuzz COMMAND expr_variable_reordering_fuzz -runs=65000)
    target_compile_options(expr_variable_reordering_fuzz PRIVATE $<$<C_COMPILER_ID:Clang>:-g -O1 -fsanitize=fuzzer>)
    target_link_libraries(expr_variable_reordering_fuzz PRIVATE $<$<C_COMPILER_ID:Clang>:-fsanitize=fuzzer> cache_fuzz_utils)
endif()





