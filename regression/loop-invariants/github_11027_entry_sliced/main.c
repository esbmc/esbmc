/* Regression: GitHub #11027 -- i enters the loop below 5, so no run from the
 * entry state reaches i == 50. The entry probe reads i's value before the
 * havoc, whose definition the main formula does not need; the slicer must
 * keep it, or the probe sees a free i and reports a false failure. */
unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n < 5);
  unsigned i = n * 3;
  __ESBMC_loop_invariant(i <= 100);
  while (i < 100)
  {
    __ESBMC_assert(i != 50, "i != 50");
    i++;
  }
}
