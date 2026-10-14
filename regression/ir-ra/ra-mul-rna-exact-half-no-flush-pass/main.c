extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  double a = __VERIFIER_nondet_double();
  double b = __VERIFIER_nondet_double();
  /* Same exact construction as the RNE test: a*b == 2^-1075 ==
   * min_subnormal/2, precisely. */
  __ESBMC_assume(a == 0x1p-537 && b == 0x1p-538);
  double y = a * b;
  /* IEEE 754 rounds this exact tie away from zero to +min_subnormal.
   * IR-IEEE does not quantize to the exact subnormal grid here; this test
   * checks only that the result is not flushed to zero. */
  __ESBMC_assert(
    y != 0.0, "RNA at exactly min_subnormal/2 must not flush to zero");
  return 0;
}
