add_library(test_util_irep util_irep.cpp)
target_link_libraries(test_util_irep PUBLIC irep2 ${Boost_LIBRARIES} bigint util_esbmc)
target_include_directories(test_util_irep
    PUBLIC ${Boost_INCLUDE_DIRS}
)
add_library(test_goto_factory goto_factory.cpp)
target_link_libraries(test_goto_factory PUBLIC irep2 clangcfrontend clangcppfrontend clibs langapi)
target_include_directories(test_goto_factory
    PUBLIC ${Boost_INCLUDE_DIRS}
)
