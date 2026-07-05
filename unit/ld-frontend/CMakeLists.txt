new_unit_test(ld_parser_test "test_parser.cpp" "ldfrontend")
target_compile_definitions(
  ld_parser_test PRIVATE FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

new_unit_test(ld_property_test "test_property.cpp" "ldfrontend")
target_compile_definitions(
  ld_property_test PRIVATE FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

new_unit_test(ld_ir_gen_test "test_ir_gen.cpp" "ldfrontend")
