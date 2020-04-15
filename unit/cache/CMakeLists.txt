# Base libraries
add_library(cache_test_utils cache_test_utils.cpp)
target_compile_definitions(cache_test_utils PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(cache_test_utils cache_methods cache_algorithms cache_containers util_esbmc bigint ${Boost_LIBRARIES})

add_library(cache_fuzz_utils cache_fuzz_utils.cpp)
target_link_libraries(cache_fuzz_utils cache_test_utils)

# LRU Cache
add_executable(lru_cache_test lru_cache.test.cpp)
target_link_libraries(lru_cache_test cache_test_utils)
add_test(NAME LruCacheTest COMMAND lru_cache_test)

# CRC Set Container Test
add_executable(crc_set_container_test crc_set_container.test.cpp)
target_link_libraries(crc_set_container_test cache_test_utils)
add_test(NAME CrcSetContainerTest COMMAND crc_set_container_test)

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

    # Expr Green Normal Form Fuzz
    add_executable(expr_green_normal_form_fuzz expr_green_normal_form.fuzz.cpp)
    add_test(NAME GreenNormalFormFuzz COMMAND expr_green_normal_form_fuzz -runs=65000)
    target_compile_options(expr_green_normal_form_fuzz PRIVATE $<$<C_COMPILER_ID:Clang>:-g -O1 -fsanitize=fuzzer>)
    target_link_libraries(expr_green_normal_form_fuzz PRIVATE $<$<C_COMPILER_ID:Clang>:-fsanitize=fuzzer> cache_fuzz_utils)
endif()





