/* The twin of all_witnesses_code_after_failed_assert: the branch restores x,
   so w holds on every path and must be reported PASSED (esbmc/esbmc#7900). */
int main()
{
  int x = 0;
  if (nondet_int())
  {
    __ESBMC_assert(0, "v");
    x = 1;
    x = x - 1;
  }
  __ESBMC_assert(x == 0, "w");
  return 0;
}
