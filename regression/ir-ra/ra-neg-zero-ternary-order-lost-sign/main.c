extern double __VERIFIER_nondet_double(void);
extern int __VERIFIER_nondet_int(void);
extern int __ESBMC_rounding_mode;

int main(void)
{
  /* KNOWNBUG: companion to ra-neg-zero-if-merge-lost-sign, pinning the
   * other branch order of the same underlying gap. z's two possible
   * values (0.0 and -0.0) come from a C ternary rather than an if/else,
   * but negative-zero metadata is still lost across the merge here: with
   * this specific branch order, ESBMC's own constant folder (which
   * treats +0.0 and -0.0 as interchangeable, per IEEE 754 ==) keeps the
   * +0.0 branch's identity. Reversing the branch order
   * (c ? -0.0 : 0.0, see ra-neg-zero-literal-div-sign's sibling
   * investigation) happens to keep the -0.0 branch's identity instead
   * and gives the correct answer by the same coincidence -- neither
   * order is a soundly-handled merge. */
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN, concrete */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > 0.0);
  int c = __VERIFIER_nondet_int();
  double z = c ? 0.0 : -0.0;
  double r = x / z;
  /* IEEE 754: when c is false, z is -0.0 and r must be -infinity, so
   * this assertion is false and should be reported as VERIFICATION
   * FAILED. It currently is not: the lost merge metadata makes z's
   * divisor sign look unconditionally positive, so r is (wrongly)
   * always +infinity. */
  __ESBMC_assert(
    r > 0.0, "ternary-merged -0.0 must still yield -infinity when divided into");
  return 0;
}
