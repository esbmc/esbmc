# SMT-backend unit tests. Each backend guards its own block on the
# canonical ENABLE_<solver> flag so the suite still builds when the user
# disables a solver.

if(ENABLE_Z3)
  # Link against `solverz3` (the static lib carrying z3_conv.cpp) rather
  # than ${Z3_LIB} directly: solverz3 already pulls in fmt::fmt, the right
  # Z3 lib, and on BUILD_STATIC the required -pthread, libgomp, -ldl. smt /
  # smttuple / smtfp / prop carry the smt_convt machinery the tests
  # instantiate via create_new_z3_solver. `bigint` is named explicitly even
  # though util_esbmc transitively pulls it in today — keeps the link line
  # correct if that transitive edge disappears.

  # mk_smt_int precision contract (issue #4642 / PR #4647).
  new_unit_test(z3_bigint_test "z3_bigint_mk_smt_int.test.cpp"
                "solverz3;smt;smttuple;smtfp;prop;util_esbmc;bigint")
  target_include_directories(z3_bigint_test
    PRIVATE ${Z3_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/solvers/z3)

  # End-to-end SMT lowering of bigint_type2t (issue #4642 PR 2C).
  new_unit_test(z3_bigint_smt_test "z3_bigint_smt.test.cpp"
                "solverz3;smt;smttuple;smtfp;prop;util_esbmc;bigint")
  target_include_directories(z3_bigint_smt_test
    PRIVATE ${Z3_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/solvers/z3)
endif()
