add_library(test_c_frontend_util util.cpp)
add_executable(typecfrontend typecast.test.cpp)
target_link_libraries(typecfrontend ${Boost_LIBRARIES} clangcfrontend test_c_frontend_util)
add_test(NAME TypecastCFrontend COMMAND typecfrontend)


if(ENABLE_FUZZER)
    add_executable(typecast_fuzz typecast.fuzz.cpp)
    add_test(NAME TypecastFuzz COMMAND typecast_fuzz -runs=6500000)
    target_compile_options(typecast_fuzz PRIVATE $<$<C_COMPILER_ID:Clang>:-g -O1 -fsanitize=fuzzer>)
    target_include_directories(typecast_fuzz PRIVATE ${MODELS_INCLUDE})
    target_link_libraries(typecast_fuzz PUBLIC $<$<C_COMPILER_ID:Clang>:-fsanitize=fuzzer> clangcfrontend test_c_frontend_util)
endif()