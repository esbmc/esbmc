add_library(util_esbmc solidity_symbol.cpp
    #c_sizeof.cpp c_link.cpp c_typecast.cpp fix_symbol.cpp
)
# Boost is needed by anything that touches irep2
target_include_directories(util_esbmc
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS}
)

target_link_libraries(util_esbmc ${Boost_LIBRARIES})
