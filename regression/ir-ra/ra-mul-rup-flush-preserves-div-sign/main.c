extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF: negative tiny denom flushes to 0 */
  double a = __VERIFIER_nondet_double();
  __ESBMC_assume(a > 0.0 && a < 1e-162);
  double denom = (-a) * a; /* flushes to zero under RUP; the true IEEE 754
                               value at this point is -0.0 */
  double z = 1.0 / denom;
  /* IEEE 754: 1.0 / -0.0 == -infinity. This regression checks that
   * IR-IEEE recovers the sign of a zero produced by flushing a negative
   * value when selecting the sign of the infinity result. */
  __ESBMC_assert(
    z < 0.0,
    "dividing by a zero flushed from a negative value must give -infinity");
  return 0;
}
