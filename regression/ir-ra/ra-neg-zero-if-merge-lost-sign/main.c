extern double __VERIFIER_nondet_double(void);
extern int __VERIFIER_nondet_int(void);
extern int __ESBMC_rounding_mode;

int main(void)
{
  /* KNOWNBUG: pins a residual gap in negative-zero tracking that is
   * distinct from, and not fixed by, the literal-constant work in this
   * suite. z's two possible values (0.0 and -0.0) are assigned in
   * separate branches of an `if`, so the SSA merge that combines them
   * goes through the generic conditional/branch-merge machinery, not
   * through mk_subnormal_flush or the constant-conversion path -- and
   * that generic merge does not forward negative-zero metadata across
   * the join. (The superficially similar ternary form `c ? -0.0 : 0.0`
   * is NOT a counterexample to this: it happens to reach the right
   * answer only because ESBMC's own constant folder treats +0.0 and
   * -0.0 as interchangeable, per IEEE 754 ==, and arbitrarily keeps one
   * branch's identity -- reordering the ternary's branches flips the
   * result. Neither form is a soundly-handled merge.) */
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN, concrete */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > 0.0);
  int c = __VERIFIER_nondet_int();
  double z = 0.0;
  if (c)
    z = -0.0;
  double r = x / z;
  /* IEEE 754: when c is true, z is -0.0 and r must be -infinity, so this
   * assertion is false and should be reported as VERIFICATION FAILED.
   * It currently is not: the lost merge metadata makes z's divisor sign
   * look unconditionally positive, so r is (wrongly) always +infinity. */
  __ESBMC_assert(r > 0.0, "if-merged -0.0 must still yield -infinity when divided into");
  return 0;
}
