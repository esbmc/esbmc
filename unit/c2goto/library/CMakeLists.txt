# Since the C models aren't really compiled we need
# to add some helpers here

set(MODELS_INCLUDE ${CMAKE_SOURCE_DIR}/src/c2goto)

new_unit_test(builtin_libstest "builtin_libs.test.cpp" "bigint")
target_include_directories(builtin_libstest PRIVATE ${MODELS_INCLUDE})

new_fuzz_test(builtin_libsfuzz "builtin_libs.fuzz.cpp" "bigint")
if(ENABLE_FUZZER)
    target_include_directories(builtin_libsfuzz PRIVATE ${MODELS_INCLUDE})
endif()