# SMT-backend unit tests. Each backend guards its own block on the
# canonical ENABLE_<solver> flag so the suite still builds when the user
# disables a solver.

if(ENABLE_Z3)
  new_unit_test(z3_bigint_test "z3_bigint_mk_smt_int.test.cpp"
                "util_esbmc;${Z3_LIB}")
  target_include_directories(z3_bigint_test PRIVATE ${Z3_INCLUDE_DIRS})
endif()
