extern double __VERIFIER_nondet_double(void);

int main(void)
{
  /* x is genuinely nondet (not pinned to a specific value), so neither
   * division below can be resolved by ESBMC's own constant folder --
   * each is only decidable via the actual SMT encoding, which exercises
   * the literal zero constants' tracked sign directly (no branching/
   * merge is needed to reach them, so this stays isolated to the
   * constant-conversion path). */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > 0.0);

  double neg = x / (-0.0);
  double pos = x / 0.0;

  /* IEEE 754: positive / -0.0 == -infinity, positive / +0.0 == +infinity.
   * Checking both in the same program confirms that recording
   * negative-zero metadata for the -0.0 literal does not also mark the
   * ordinary +0.0 literal as negative. */
  __ESBMC_assert(
    neg < 0.0, "dividing a positive value by literal -0.0 must give -infinity");
  __ESBMC_assert(
    pos > 0.0, "dividing a positive value by literal +0.0 must give +infinity");
  return 0;
}
