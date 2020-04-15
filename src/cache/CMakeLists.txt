add_subdirectory(algorithms)
add_subdirectory(containers)
add_subdirectory(cache)

add_library(smt_cache
        cache.h)

target_link_libraries(smt_cache  cache_methods cache_algorithms cache_containers )