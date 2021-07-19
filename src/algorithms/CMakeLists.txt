add_library(goto_algorithms goto_algorithms.cpp)
target_link_libraries(goto_algorithms gotoprograms)
target_include_directories(goto_algorithms
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS}
)