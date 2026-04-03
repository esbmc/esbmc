/* Regression test for https://github.com/esbmc/esbmc/issues/3936
 * False plain invariant (base case): idx starts at 0, but the invariant
 * claims idx > 0.  The base-case ASSERT should fail immediately.
 *
 * VERIFICATION FAILED is expected.
 */
int main()
{
  int idx_v0;
  int idx;

  __ESBMC_loop_invariant(idx > 0); /* false at base: idx = 0 initially */
  __ESBMC_loop_invariant(
    __ESBMC_forall(&idx_v0,
      !(0 <= idx_v0) || !(idx_v0 < idx) || (idx_v0 < 3)));

  for (idx = 0; idx < 3; ++idx)
  {
    __ESBMC_assert(0 <= idx, "bounds");
  }

  return 0;
}
