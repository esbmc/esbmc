add_subdirectory(algorithms)
add_library(smt_cache
        algorithms
        algorithms/ssa_step_algorithm_debug.cpp
        green/green_cache.cpp
        algorithms/green_normal_form.cpp algorithms/ssa_flattener.cpp)

target_link_libraries(smt_cache cache_algorithms)