unsigned nondet_uint(void);

int main(void)
{
  unsigned x = nondet_uint();
  __ESBMC_assume(x != 0u);
  unsigned y = -x;

  /* False for every x != 0, so the checker must report it: a negation that
   * collapsed to an unconstrained value would let this through. */
  __ESBMC_assert(y == 0u, "-x is never zero for x != 0");
  return 0;
}
