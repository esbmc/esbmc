/* Regression test for https://github.com/esbmc/esbmc/issues/3936
 * Variant: quantifier invariant declared FIRST, plain invariant SECOND.
 * This tests that the fix collects all LOOP_INVARIANT instructions
 * regardless of source order.
 *
 * VERIFICATION SUCCESSFUL is expected.
 */
int main()
{
  int idx_v0;
  int idx;

  /* Quantifier invariant comes first in source order (appears further from
   * the loop head in GOTO; the plain invariant is closer). */
  __ESBMC_loop_invariant(
    __ESBMC_forall(&idx_v0,
      !(0 <= idx_v0) || !(idx_v0 < idx) || (idx_v0 < 3)));
  __ESBMC_loop_invariant(0 <= idx);

  for (idx = 0; idx < 3; ++idx)
  {
    __ESBMC_assert(0 <= idx, "bounds");
  }

  return 0;
}
