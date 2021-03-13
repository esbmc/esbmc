add_library(test_util_irep util_irep.cpp)
target_link_libraries(test_util_irep ${Boost_LIBRARIES} bigint util_esbmc)