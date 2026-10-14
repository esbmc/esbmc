extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > 0.0 && x < 1e-162);
  double y = x * x; /* exact product is strictly positive and strictly
                        below min_subnormal (~4.94e-324) in magnitude */
  /* Under round-toward-+infinity, a strictly positive exact result can
   * never round down to 0.0 -- RUP always rounds a positive
   * non-representable value UP to the next representable value. */
  __ESBMC_assert(y != 0.0, "RUP must not flush a positive tiny product to zero");
  return 0;
}
