add_library(test_util_irep util_irep.cpp)
target_link_libraries(test_util_irep PUBLIC util_esbmc)

# OBJECT library so mode_table.o is always injected directly into the
# final link (not archived), guaranteeing mode_table is defined before
# langapi/mode.cpp.o looks for it.
add_library(mode_table_obj OBJECT mode_table.cpp)
target_link_libraries(mode_table_obj PRIVATE langapi)

add_library(test_goto_factory goto_factory.cpp)
target_link_libraries(test_goto_factory PUBLIC irep2 clangcfrontend clangcppfrontend clibs)
target_include_directories(test_goto_factory
    PUBLIC ${Boost_INCLUDE_DIRS}
)
