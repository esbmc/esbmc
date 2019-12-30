add_executable(biginttest bigint.test.cpp)
target_link_libraries(biginttest ${Boost_LIBRARIES} bigint)

add_test(NAME BigInt COMMAND biginttest) 