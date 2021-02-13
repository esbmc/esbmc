add_library(cache_algorithms
        expr_variable_reordering.cpp
        expr_green_normal_form.cpp)

# Boost is needed by anything that touches irep2
target_include_directories(cache_algorithms
        PRIVATE ${Boost_INCLUDE_DIRS})