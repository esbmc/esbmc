add_library(prop pointer_logic.cpp)
target_link_libraries(prop PUBLIC esbmc_common PRIVATE fmt::fmt)
target_include_directories(prop
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
