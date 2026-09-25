extern double __VERIFIER_nondet_double(void);

int main(void)
{
  /* Default rounding mode: round-to-nearest-even (RNE). */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > 1.9e-162 && x < 2.0e-162);
  double y = x * x; /* exact product lands strictly between
                        min_subnormal/2 (~2.47e-324) and min_subnormal
                        (~4.94e-324) */
  /* IEEE 754 ties-to-even rounds any value strictly above the
   * half-subnormal boundary up to +min_subnormal, never down to 0.
   * IR-IEEE does not quantize to the exact subnormal grid here; this test
   * checks only that the result is not flushed to zero. */
  __ESBMC_assert(
    y != 0.0, "RNE above the half-subnormal boundary must not flush to zero");
  return 0;
}
