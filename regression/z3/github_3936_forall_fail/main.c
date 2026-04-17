/* Regression test for https://github.com/esbmc/esbmc/issues/3936
 * False forall invariant: the quantifier claims that every index seen so
 * far satisfies idx_v0 < 0, which is impossible for non-negative indices.
 * The base-case ASSERT for the forall should fail.
 *
 * VERIFICATION FAILED is expected.
 */
int main()
{
  int idx_v0;
  int idx;

  __ESBMC_loop_invariant(0 <= idx);
  __ESBMC_loop_invariant(
    __ESBMC_forall(&idx_v0,
      !(0 <= idx_v0) || !(idx_v0 < idx) || (idx_v0 < 0))); /* idx_v0 < 0 is always false */

  for (idx = 0; idx < 3; ++idx)
  {
    __ESBMC_assert(0 <= idx, "bounds");
  }

  return 0;
}
