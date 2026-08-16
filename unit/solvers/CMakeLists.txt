# SMT-backend unit tests. Each backend guards its own block on the
# canonical ENABLE_<solver> flag so the suite still builds when the user
# disables a solver.

if(ENABLE_Z3)
  # Link against `solverz3` (the static lib carrying z3_conv.cpp) rather
  # than ${Z3_LIB} directly: solverz3 already pulls in fmt::fmt, the right
  # Z3 lib, and on BUILD_STATIC the required -pthread, libgomp, -ldl. This
  # also makes z3_convt and create_new_z3_solver available so the test can
  # exercise ESBMC's overload (not just Z3's underlying numeral API).
  # `bigint` is named explicitly even though util_esbmc transitively pulls
  # it in today — keeps the link line correct if that transitive edge
  # disappears.
  new_unit_test(z3_bigint_test "z3_bigint_mk_smt_int.test.cpp"
                "solverz3;smt;smttuple;smtfp;prop;util_esbmc;bigint")
  target_include_directories(z3_bigint_test
    PRIVATE ${Z3_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/solvers/z3)

  # Drives array_convt's bounded select/store directly: the boundary they guard
  # is unreachable from C input, so only a unit test can pin it. Z3 serves
  # merely as a concrete backend for the flattener to sit on.
  new_unit_test(array_conv_bounded_index_test
                "array_conv_bounded_index.test.cpp"
                "solverz3;smt;smttuple;smtfp;prop;util_esbmc;bigint")
  target_include_directories(array_conv_bounded_index_test
    PRIVATE ${Z3_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/solvers/z3)
endif()

if(ENABLE_SMTLIB)
  # Verdict scanner shared by the one-shot subprocess backends (bitwuzllob,
  # neurosym); lives in the smtlib library. util_esbmc supplies the logging
  # and filesystem symbols the rest of the translation unit pulls in.
  new_unit_test(oneshot_verdict_test "oneshot_verdict.test.cpp"
                "smtlib;filesystem;util_esbmc;fmt::fmt")
endif()

# The SMT equivalence checker for simplifier rewrites (#4625). The installed
# checker aborts on a mismatch, so only a direct call can assert that it
# detects one. Any built-in solver serves; `solve` picks the default.
# Links the `solvers` interface target rather than a named backend: solve.cpp
# registers every backend compiled in, so the creator symbols must all be
# present. Whichever is the default answers the queries.
new_unit_test(simplification_equivalence_test
              "simplification_equivalence.test.cpp"
              "solvers;util_esbmc;filesystem;bigint")
