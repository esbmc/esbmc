// Under --smt-formula-only every claim returns P_SMTLIB, so neither assertion
// is ever solved. multi_property_check used to leave final_result at its
// P_UNSATISFIABLE seed, closing the run VERIFICATION SUCCESSFUL and exit 0
// over an analysis that never ran — even though "b" is violable (issue #5934).
int main(void)
{
  int x;
  __ESBMC_assume(x > 0);
  __ESBMC_assert(x > 0, "a");
  __ESBMC_assert(x > 1, "b");
  return 0;
}
