add_executable(ld_unit_parser test_parser.cpp)

target_include_directories(ld_unit_parser
  PRIVATE ${CMAKE_SOURCE_DIR}/src
  PRIVATE ${CMAKE_BINARY_DIR}/src
)

target_link_libraries(ld_unit_parser
  PRIVATE ldfrontend util_esbmc irep2 langapi ${Boost_LIBRARIES}
)

target_compile_definitions(ld_unit_parser
  PRIVATE FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

add_test(NAME ld_unit_parser COMMAND ld_unit_parser)
