/* --multi-property --loop-invariant with an incomplete invariant that leaves
 * the post-loop assertion unprovable by k-induction.
 *
 * The loop invariants cover the ranges of sum, bound2, and idx2, but omit
 * the relational invariant (sum % bound2 == idx2 % bound2).  Without it the
 * inductive step cannot discharge the post-loop assertion, so the result is
 * VERIFICATION UNKNOWN.
 *
 * When UNKNOWN is reached, ESBMC should run a per-VCC diagnostic pass and
 * identify the specific unresolved property by name.
 *
 * Regression test for GitHub issue #4022.
 */

int main()
{
  int bound2, idx2, sum;
  idx2 = 0;
  __ESBMC_assume(0 <= sum && sum < 256);
  __ESBMC_assume(4 <= bound2 && bound2 < 256);
  __ESBMC_assume(0 <= idx2 && idx2 <= bound2);
  __ESBMC_assume(256 % bound2 == 0);
  __ESBMC_assume((sum % bound2) == idx2 % bound2);

  /* Range invariants only — relational invariant intentionally omitted */
  __ESBMC_loop_invariant(0 <= sum && sum < 256);
  __ESBMC_loop_invariant(4 <= bound2 && bound2 < 256);
  __ESBMC_loop_invariant(0 <= idx2 && idx2 <= bound2);
  __ESBMC_loop_invariant(256 % bound2 == 0);

  while(idx2 < bound2)
  {
    sum = (sum + 1) % 256;
    ++idx2;
  }
  __ESBMC_assert((sum % bound2) == (idx2 % bound2), "post loop modular equality");
  return 0;
}
