/* Regression test for https://github.com/esbmc/esbmc/issues/3936
 * Sanity check: quantifier invariant with no assertion inside the loop.
 * This case passed before the fix and must continue to pass.
 *
 * VERIFICATION SUCCESSFUL is expected.
 */
int main()
{
  int vec[3];
  int sum = 0;
  int idx_v0;
  int idx;

  __ESBMC_loop_invariant(0 <= idx);
  __ESBMC_loop_invariant(
    __ESBMC_forall(&idx_v0, (4 <= idx_v0) || (idx_v0 <= 4)));

  for (idx = 0; idx < 3; ++idx)
  {
    sum += idx;
  }

  return 0;
}
