/* esbmc/esbmc#7900: --all-witnesses keeps checking claims past a violation,
   but remove_unreachable still ran for it and deleted the code after the
   failed assertion, so w was reported PASSED. */
int main()
{
  int x = 0;
  if (nondet_int())
  {
    __ESBMC_assert(0, "v");
    x = 1;
    x = x + 1;
  }
  __ESBMC_assert(x == 0, "w");
  return 0;
}
