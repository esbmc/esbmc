// Same program as github_5934, but --smt-formula-too emits the formula *and*
// solves every claim, so no claim returns P_SMTLIB and the non-verdict path
// is never entered. Pins that dumping the formula under --multi-property
// still reports the violation of "b" (issue #5934).
int main(void)
{
  int x;
  __ESBMC_assume(x > 0);
  __ESBMC_assert(x > 0, "a");
  __ESBMC_assert(x > 1, "b");
  return 0;
}
