add_library(goto2lean goto2lean.cpp)

target_include_directories(goto2lean
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(goto2lean PUBLIC esbmc_common gotoprograms)
