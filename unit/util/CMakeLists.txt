add_executable(lrucachetest lru_cache.test.cpp)
target_compile_definitions(lrucachetest PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(lrucachetest ${Boost_LIBRARIES})
add_test(NAME LruCache COMMAND lrucachetest)
