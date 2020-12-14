# Since the C models aren't really compiled we need
# to add some helpers here

set(MODELS_INCLUDE ${CMAKE_SOURCE_DIR}/src/c2goto/headers)
set(MODELS_LIB ${CMAKE_SOURCE_DIR}/src/c2goto/library)

add_executable(builtin_libs_test builtin_libs.test.cpp)
target_include_directories(builtin_libs_test PRIVATE ${MODELS_INCLUDE})
target_link_libraries(builtin_libs_test PUBLIC ${MODELS_LIB}/builtin_libs.c ${Boost_LIBRARIES})

add_test(NAME BuiltinLibsModelTest COMMAND builtin_libs_test)

if(NOT BUILD_STATIC)
    add_definitions(-DBOOST_TEST_DYN_LINK)
endif()


if(ENABLE_FUZZER)
    add_executable(builtin_libs_fuzz builtin_libs.fuzz.cpp)
    add_test(NAME BuiltinModelFuzz COMMAND builtin_libs_fuzz -runs=6500000)
    target_compile_options(builtin_libs_fuzz PRIVATE $<$<C_COMPILER_ID:Clang>:-g -O1 -fsanitize=fuzzer>)
    target_include_directories(builtin_libs_fuzz PRIVATE ${MODELS_INCLUDE})
    target_link_libraries(builtin_libs_fuzz PUBLIC $<$<C_COMPILER_ID:Clang>:-fsanitize=fuzzer> ${MODELS_LIB}/builtin_libs.c)
endif()