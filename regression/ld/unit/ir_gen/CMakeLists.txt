add_executable(ld_unit_ir_gen test_ir_gen.cpp)

target_include_directories(ld_unit_ir_gen
  PRIVATE ${CMAKE_SOURCE_DIR}/src
  PRIVATE ${CMAKE_BINARY_DIR}/src
)

target_link_libraries(ld_unit_ir_gen
  PRIVATE ldfrontend util_esbmc irep2 langapi ${Boost_LIBRARIES}
)

add_test(NAME ld_unit_ir_gen COMMAND ld_unit_ir_gen)
