add_library(cache_methods
        green_cache.cpp)

set_target_properties(cache_methods PROPERTIES
        CXX_STANDARD 17)

# Boost is needed by anything that touches irep2
target_include_directories(cache_methods
        PRIVATE ${Boost_INCLUDE_DIRS})