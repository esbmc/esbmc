extern double __VERIFIER_nondet_double(void);

int main(void)
{
  /* Default rounding mode: round-to-nearest-even (RNE). */
  double a = __VERIFIER_nondet_double();
  double b = __VERIFIER_nondet_double();
  /* 0x1p-537 and 0x1p-538 are exact powers of two, each an ordinary
   * (non-subnormal) representable double, so their product is computed
   * exactly: 2^-537 * 2^-538 == 2^-1075 == min_subnormal/2, precisely. */
  __ESBMC_assume(a == 0x1p-537 && b == 0x1p-538);
  double y = a * b;
  /* Ties-to-even: exactly at the half-subnormal boundary rounds to zero
   * (0 is the "even" candidate). */
  __ESBMC_assert(y == 0.0, "RNE at exactly min_subnormal/2 must round to zero");
  return 0;
}
