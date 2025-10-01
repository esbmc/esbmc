if (NOT WIN32)
set(BUILDID_DIR "${CMAKE_BINARY_DIR}/src/esbmc")
set(BUILDID_TXT "${BUILDID_DIR}/buildidobj.txt")
set(BUILDID_C   "${BUILDID_DIR}/buildidobj.c")

# Tell CMake this file is generated so it won't try to stat it during add_executable()
set_source_files_properties("${BUILDID_C}" PROPERTIES GENERATED TRUE)
add_custom_target(gen-buildid DEPENDS "${BUILDID_C}")

new_unit_test(esbmc_parseoption_test "esbmc_parseoptions_test.cpp"
              "util_esbmc;bigint;langapi;fmt::fmt;yaml-cpp::yaml-cpp;nlohmann_json::nlohmann_json;pythonfrontend;gotoalgorithms")

add_dependencies(esbmc_parseoption_test gen-buildid)

target_sources(esbmc_parseoption_test PRIVATE
  bmc_dummy.cpp
  goto2c_check_dummy.cpp
  ${CMAKE_SOURCE_DIR}/src/esbmc/options.cpp
  ${CMAKE_SOURCE_DIR}/src/esbmc/show_vcc.cpp
  ${CMAKE_SOURCE_DIR}/src/esbmc/esbmc_parseoptions.cpp
  ${CMAKE_SOURCE_DIR}/src/c2goto/cprover_blank_library.cpp
  ${BUILDID_C}
)
endif()