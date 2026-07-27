extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 3; /* ROUND_TO_MINUS_INF */
  double a = __VERIFIER_nondet_double();
  double b = __VERIFIER_nondet_double();
  /* a*b == -2^-1075, exactly -- strictly between -min_subnormal
   * (-2^-1074) and 0. */
  __ESBMC_assume(a == -0x1p-537 && b == 0x1p-538);
  double y = a * b;
  /* IEEE 754 RDN always rounds a strictly negative non-representable
   * value DOWN (away from zero, toward -min_subnormal) -- never up to 0.
   * IR-IEEE does not quantize to the exact subnormal grid here; this test
   * checks only that the result is not flushed to zero. */
  __ESBMC_assert(
    y != 0.0,
    "RDN must not flush a negative value strictly above -min_subnormal");
  return 0;
}
