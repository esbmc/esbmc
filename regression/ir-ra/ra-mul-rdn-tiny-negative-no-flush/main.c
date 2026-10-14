extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 3; /* ROUND_TO_MINUS_INF */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > 0.0 && x < 1e-162);
  double y = (-x) * x; /* exact product is negative, strictly below
                           min_subnormal in magnitude (~4.94e-324) */
  /* Under round-toward-minus-infinity, a strictly negative exact result
   * can never round up to 0.0 -- RDN always rounds a negative
   * non-representable value DOWN (further from zero). */
  __ESBMC_assert(y != 0.0, "RDN must not flush a negative tiny product to zero");
  return 0;
}
