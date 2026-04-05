/* Regression test for https://github.com/esbmc/esbmc/issues/3936
 * True failure case: the assertion inside the loop is genuinely wrong.
 * idx starts at 1, so the forall invariant claims all seen indices were >= 1,
 * but the assertion asserts idx > 3 which never holds for idx in [1,3).
 *
 * This ensures the fix does not suppress real failures.
 *
 * VERIFICATION FAILED is expected.
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
    __ESBMC_assert(idx > 3, "this is genuinely false");
  }

  return 0;
}
