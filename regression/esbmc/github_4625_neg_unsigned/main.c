unsigned nondet_uint(void);

int main(void)
{
  unsigned x = nondet_uint();
  unsigned y = -x;

  /* Modular negation: both hold for every unsigned x. Simplifying -x to
   * (2^32 - x) % 2^32 left `(0 - x) % 0` in the SSA, because 2^32 is not
   * representable in the 32-bit type the constant was given. */
  __ESBMC_assert((unsigned)(x + y) == 0u, "x + (-x) == 0");
  __ESBMC_assert(x != 0u || y == 0u, "-0 == 0");
  return 0;
}
