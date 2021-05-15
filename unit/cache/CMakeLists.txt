# Base libraries
add_library(cache_test_utils cache_test_utils.cpp)
target_link_libraries(cache_test_utils cache_methods cache_algorithms cache_containers util_esbmc bigint)

add_library(cache_fuzz_utils cache_fuzz_utils.cpp)
target_link_libraries(cache_fuzz_utils cache_test_utils)

# LRU Cache
new_unit_test(lru_cache_test "lru_cache.test.cpp" "cache_test_utils")

# CRC Set Container Test
new_unit_test(crc_set_container_test "crc_set_container.test.cpp" "cache_test_utils")

# Expr Reordering Test
new_unit_test(expr_variable_reordering_test "expr_variable_reordering.test.cpp" "cache_test_utils")
new_fuzz_test(expr_variable_reordering_fuzz "expr_variable_reordering.fuzz.cpp" "cache_fuzz_utils")

# Expr Green Normal Form Test
new_unit_test(expr_green_normal_form_test "expr_green_normal_form.test.cpp" "cache_test_utils")
new_fuzz_test(expr_gree_normal_form_fuzz "expr_green_normal_form.fuzz.cpp" "cache_fuzz_utils")

# Cache Fuzz Utils Test
new_unit_test(cache_fuzz_utils_test "cache_fuzz_utils.test.cpp" "cache_fuzz_utils")
