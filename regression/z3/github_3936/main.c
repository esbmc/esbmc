/* Regression test for https://github.com/esbmc/esbmc/issues/3936
 * Spurious counterexample when combining a quantifier invariant with an
 * assertion inside the loop body (the failing case from the issue report).
 *
 * extract_invariants_near() used to stop at the first (closest) LOOP_INVARIANT
 * instruction due to an early break. When two __ESBMC_loop_invariant() calls
 * are present, only the forall invariant was collected; the plain (0 <= idx)
 * invariant was silently dropped. As a result, after the havoc step the tool
 * did not ASSUME (0 <= idx), so idx could be INT_MIN in the inductive step,
 * causing the assertion inside the loop to fail spuriously.
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
    __ESBMC_assert(0 <= idx, "bounds");
  }

  return 0;
}
