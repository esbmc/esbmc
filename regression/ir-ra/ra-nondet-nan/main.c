extern double __VERIFIER_nondet_double(void);

int main(void)
{
  /* A nondet double may be NaN.  An unconstrained nondet double that equals
   * itself (which NaN does not) must not be assumed without justification. */
  double x = __VERIFIER_nondet_double();

  __ESBMC_assert(x == x, "unconstrained nondet double may be NaN");
  return 0;
}
