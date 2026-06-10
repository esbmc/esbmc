include(FetchContent)

fetchcontent_declare(
  pugixml
  GIT_REPOSITORY https://github.com/zeux/pugixml.git
  GIT_TAG        v1.14
)
fetchcontent_makeavailable(pugixml)

add_library(ldfrontend STATIC
  ld_language.cpp
  parser/plcopen_xml_parser.cpp
  parser/ld_ast.cpp
  semantics/sos_semantics.cpp
  semantics/type_checker.cpp
  ir/ld_ir.cpp
  ir/ld_ir_builder.cpp
  ir_gen/ld_converter.cpp
  property/yaml_property_parser.cpp
  property/property_encoder.cpp
  verify/ld_verify.cpp
)

target_include_directories(ldfrontend
  PUBLIC  ${CMAKE_SOURCE_DIR}/src
          ${Boost_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# ld_verify.cpp drives the esbmc binary via Boost.Process and locates it via
# Boost.DLL, which need Boost.Filesystem/System at link time.
target_link_libraries(ldfrontend
  PUBLIC pugixml::static util_esbmc irep2
  PRIVATE fmt::fmt yaml-cpp::yaml-cpp ${Boost_LIBRARIES}
)
