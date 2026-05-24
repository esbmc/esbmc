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
endif()
