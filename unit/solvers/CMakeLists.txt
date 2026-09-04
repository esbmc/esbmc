# SMT-backend unit tests. Each backend guards its own block on the
# canonical ENABLE_<solver> flag so the suite still builds when the user
# disables a solver.

if(ENABLE_Z3)
  # Link against the `solvers` interface lib (pulls in solve + smt + prop)
  # plus solvercamada (the static lib carrying camada_conv.cpp which hosts
  # create_new_z3_solver). `bigint` is named explicitly even
  # though util_esbmc transitively pulls it in today — keeps the link line
  # correct if that transitive edge disappears.
  new_unit_test(z3_bigint_test "z3_bigint_mk_smt_int.test.cpp"
                "solvers;util_esbmc;bigint")
endif()

# The one-shot verdict scanner's contract is camada's now
# (camada::parseOneShotVerdictLine, covered by camada's own regression suite),
# so the test that pinned ESBMC's copy went with the smtlib backend -- and with
# it upstream's oneshot_verdict_test, which linked that library.

# The array and tuple flatteners went with fp_convt: camada encodes both
# itself, natively or lowered (ArrayEncoding::Ackermann, TupleEncoding::Camada),
# so the test that pinned array_convt's bounded select/store went with it.

# The SMT equivalence checker for simplifier rewrites (#4625). The installed
# checker aborts on a mismatch, so only a direct call can assert that it
# detects one. Any built-in solver serves; `solve` picks the default. After the
# solvers flatten there is a single `solvers` library rather than an interface
# target over per-backend ones, but the link line is unchanged: camada_conv
# registers every backend camada was built with.
new_unit_test(simplification_equivalence_test
              "simplification_equivalence.test.cpp"
              "solvers;util_esbmc;filesystem;bigint")

# The node tuple flattener's update(), driven directly: the field-preservation
# contract it broke is unreachable from C input, so no regression test states
# it. `solvers` rather than a named backend -- whichever is built in answers.
new_unit_test(tuple_node_update_test "tuple_node_update.test.cpp"
              "solvers;util_esbmc;filesystem;bigint")
