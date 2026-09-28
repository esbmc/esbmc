add_library(goto2c goto2c_preprocess.cpp goto2c_check.cpp goto2c_translate.cpp expr2c.cpp)

target_include_directories(goto2c
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CLANG_INCLUDE_DIRS}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(goto2c PUBLIC esbmc_common gotoprograms ${cheaders_lib} ${ESBMC_CLANG_LIBS})
