add_executable(ld-verify main.cpp)

target_include_directories(ld-verify
  PRIVATE ${CMAKE_SOURCE_DIR}/src
  PRIVATE ${CMAKE_BINARY_DIR}/src
)

# Link only the LD frontend and core ESBMC libraries needed for parse + show_parse.
# Full verification (typecheck + symex) requires linking the full ESBMC driver,
# which is handled by invoking the esbmc binary rather than linking it here.
target_link_libraries(ld-verify
  PRIVATE ldfrontend util_esbmc irep2 langapi
          ${Boost_LIBRARIES}
)

if(BUILD_TESTING)
  set(_ld_reg ${CMAKE_SOURCE_DIR}/regression/ld)

  # ld-verify locates esbmc via $ESBMC; point it at the freshly built binary.
  # PASS_REGULAR_EXPRESSION matches the JSON report and ignores the exit code,
  # so VIOLATION (exit 10) and ERROR (exit 2) cases register as passes.
  function(add_ld_verify_test name regex)
    add_test(NAME ld-verify/${name}
             COMMAND ld-verify ${ARGN})
    set_tests_properties(ld-verify/${name} PROPERTIES
      PASS_REGULAR_EXPRESSION "${regex}"
      ENVIRONMENT "ESBMC=$<TARGET_FILE:esbmc>"
      TIMEOUT 120
      LABELS "ld-verify")
  endfunction()

  # esbmc routes by extension, so stage a .xml copy to exercise the temp-.ld path.
  configure_file(${_ld_reg}/motor_interlock/motor_interlock.ld
                 ${CMAKE_CURRENT_BINARY_DIR}/motor_interlock.xml COPYONLY)

  add_ld_verify_test(show-parse "Network 'main'"
    ${_ld_reg}/motor_interlock/motor_interlock.ld --show-parse)
  add_ld_verify_test(safe "\"result\": \"SAFE\""
    ${_ld_reg}/motor_interlock/motor_interlock.ld
    --props ${_ld_reg}/motor_interlock/props.yaml)
  add_ld_verify_test(xml-input "\"result\": \"SAFE\""
    ${CMAKE_CURRENT_BINARY_DIR}/motor_interlock.xml
    --props ${_ld_reg}/motor_interlock/props.yaml)
  add_ld_verify_test(violation "\"result\": \"VIOLATION\""
    ${_ld_reg}/mutual_exclusion_unsafe/unsafe.ld
    --props ${_ld_reg}/mutual_exclusion_unsafe/props.yaml --strategy bmc --unwind 2)
  add_ld_verify_test(incomplete "\"result\": \"INCOMPLETE\""
    ${_ld_reg}/motor_interlock/motor_interlock.ld
    --props ${_ld_reg}/motor_interlock/props.yaml --strategy bmc --unwind 2)
  add_ld_verify_test(fault-injection "\"result\": \"VIOLATION\""
    ${_ld_reg}/fault_injection_unsafe/guard.ld
    --props ${_ld_reg}/fault_injection_unsafe/props.yaml
    --strategy bmc --unwind 2 --fault-injection)
  add_ld_verify_test(unsupported-strategy "unsupported strategy"
    ${_ld_reg}/motor_interlock/motor_interlock.ld --strategy portfolio)

  # Graphical PLCopen through the runner: resolving blocks, edge contacts,
  # parallel paths and feedback all have to survive the staging path too.
  add_ld_verify_test(graphical "\"result\": \"SAFE\""
    ${_ld_reg}/stairs_light_safe/stairs_light.ld
    --props ${_ld_reg}/stairs_light_safe/props.yaml)

  # The violated property must reach the report by name, which is what makes
  # the JSON readable without the raw counterexample.
  add_ld_verify_test(violation-named
    "\"description\": \"ESD valve must be closed when manual ESD is triggered\""
    ${_ld_reg}/esd_manual_reset_fail/esd.ld
    --props ${_ld_reg}/esd_manual_reset_fail/props.yaml)

  # Anchored on "description" so the assertion cannot be met by the raw_output
  # echo, which master already emitted.
  add_ld_verify_test(error-unsupported-body
    "\"description\": \"UnsupportedConstruct\\(ST body of POU"
    ${_ld_reg}/body_st_unsupported_fail/st.ld
    --props ${_ld_reg}/body_st_unsupported_fail/props.yaml)

  # An input esbmc cannot parse yields no verdict line; that must surface as
  # ERROR rather than being folded into UNKNOWN.
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/malformed.ld "not a plcopen document")
  add_ld_verify_test(error-no-verdict
    "\"description\": \"[^\"]*No document element found"
    ${CMAKE_CURRENT_BINARY_DIR}/malformed.ld)
endif()
