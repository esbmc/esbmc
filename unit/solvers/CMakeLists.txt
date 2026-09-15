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

if(ENABLE_BITWUZLA)
  # Links `solverbitw` (the static lib carrying bitwuzla_conv.cpp) so the test
  # can construct the backend directly: mk_*_sort is not reachable through
  # smt_convt, and the sort cache it pins is a property of the backend, not of
  # any query it answers. `solvers` supplies create_solver for the model half.
  # Only the backend's own source dir is named here. Bitwuzla's include dirs
  # arrive as a usage requirement of PkgConfig::Bitwuzla, which solverbitw
  # links with the plain signature; naming that imported target here instead
  # is a configure error, as pkg_check_modules created it in the solver's
  # directory scope rather than GLOBAL.
  new_unit_test(bitwuzla_sort_cache_test "bitwuzla_sort_cache.test.cpp"
                "solvers;util_esbmc;filesystem;bigint")
  target_include_directories(bitwuzla_sort_cache_test
    PRIVATE ${CMAKE_SOURCE_DIR}/src/solvers/bitwuzla)
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

# The node tuple flattener's update(), driven directly: the field-preservation
# contract it broke is unreachable from C input, so no regression test states
# it. `solvers` rather than a named backend -- whichever is built in answers.
new_unit_test(tuple_node_update_test "tuple_node_update.test.cpp"
              "solvers;util_esbmc;filesystem;bigint")
