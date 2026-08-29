extern double __VERIFIER_nondet_double(void);

int main(void)
{
  /* Failing companion to ra-neg-zero-literal-div-sign: x is genuinely
   * nondet, so the division cannot be resolved by ESBMC's own constant
   * folder and must pass through the actual SMT encoding. */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > 0.0);

  double neg = x / (-0.0);

  /* IEEE 754: positive / -0.0 == -infinity, so this assertion is false
   * and must be reported as VERIFICATION FAILED. If the negative-zero
   * metadata were ever dropped (e.g. a future change silently treats a
   * literal -0.0 as +0.0), this assertion would incorrectly become
   * true and the test would flip to VERIFICATION SUCCESSFUL -- catching
   * the regression in the opposite direction from the passing
   * companion. */
  __ESBMC_assert(
    neg > 0.0,
    "dividing a positive value by literal -0.0 must not give +infinity");
  return 0;
}
