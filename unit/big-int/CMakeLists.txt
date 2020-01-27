add_executable(biginttest bigint.test.cpp)
target_compile_definitions(biginttest PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(biginttest ${Boost_LIBRARIES} bigint)

add_test(NAME BigInt COMMAND biginttest)



if(ENABLE_FUZZER)
    add_executable(bigintfuzz bigint.fuzz.cpp)
    add_test(NAME BigIntFuzz COMMAND bigintfuzz -runs=6500000)
    target_compile_options(bigintfuzz PRIVATE $<$<C_COMPILER_ID:Clang>:-g -O1 -fsanitize=fuzzer>)
    target_link_libraries(bigintfuzz PRIVATE $<$<C_COMPILER_ID:Clang>:-fsanitize=fuzzer> bigint)
endif()
