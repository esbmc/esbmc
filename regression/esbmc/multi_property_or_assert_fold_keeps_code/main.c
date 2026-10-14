/* The (c) || (assert(0), 0) fold erased x = 1 (esbmc/esbmc#7900). */
int main()
{
  int x = 0;
  if (nondet_int())
  {
    __ESBMC_assert(0, "v");
    x = 1;
  }
  __ESBMC_assert(x == 0, "w");
  return 0;
}
