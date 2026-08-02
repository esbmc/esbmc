# SMT-backend unit tests. Each backend guards its own block on the
# canonical ENABLE_<solver> flag so the suite still builds when the user
# disables a solver.

if(ENABLE_Z3)
  # Link against the `solvers` interface lib (pulls in solve + smt + smttuple
  # + smtfp + prop) plus solvercamada (the static lib carrying camada_conv.cpp
  # which hosts create_new_z3_solver). `bigint` is named explicitly even
  # though util_esbmc transitively pulls it in today — keeps the link line
  # correct if that transitive edge disappears.
  new_unit_test(z3_bigint_test "z3_bigint_mk_smt_int.test.cpp"
                "solvers;util_esbmc;bigint")

  # Drives array_convt's bounded select/store directly: the boundary they guard
  # is unreachable from C input, so only a unit test can pin it. Z3 serves
  # merely as a concrete backend for the flattener to sit on.
  new_unit_test(array_conv_bounded_index_test
                "array_conv_bounded_index.test.cpp"
                "solvers;util_esbmc;bigint")
endif()

if(ENABLE_SMTLIB)
  # Verdict scanner shared by the one-shot subprocess backends (bitwuzllob,
  # neurosym); lives in the smtlib library. util_esbmc supplies the logging
  # and filesystem symbols the rest of the translation unit pulls in.
  new_unit_test(oneshot_verdict_test "oneshot_verdict.test.cpp"
                "smtlib;filesystem;util_esbmc;fmt::fmt")
endif()
