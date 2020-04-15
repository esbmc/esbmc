add_library(cache_containers
        crc_set_container.cpp)

# Boost is needed by anything that touches irep2
target_include_directories(cache_containers
        PRIVATE ${Boost_INCLUDE_DIRS})