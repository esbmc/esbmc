extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF: negative tiny values flush to 0 */
  double a = __VERIFIER_nondet_double();
  __ESBMC_assume(a > 0.0 && a < 1e-162);
  double y = (-a) * a; /* exact product is negative and strictly below
                           min_subnormal in magnitude, so it flushes to
                           zero under RUP */
  /* IEEE 754 gives -0.0 here (sign bit set): a negative exact result that
   * underflows keeps its sign even when rounded to zero. This regression
   * checks that IR-IEEE preserves that sign through its side metadata. */
  __ESBMC_assert(
    __signbit(y) != 0,
    "a negative result flushed to zero must still report a negative sign bit");
  return 0;
}
