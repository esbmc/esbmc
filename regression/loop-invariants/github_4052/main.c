/* Regression test for GitHub issue #4052.
 *
 * Post-loop assertions were not individually reported when using
 * --multi-property with --loop-invariant in the inductive step.
 * The fix extends multi_property_check to cover the inductive step so
 * each assertion is solved and printed separately.
 *
 * All post-loop assertions should be proven and appear as PASSED.
 * Expected: VERIFICATION SUCCESSFUL.
 */

int main()
{
  int idx;
  int qa;
  qa %= 256;
  if (qa < 0)
    qa *= -1;
  if (qa % 2 == 0)
    ++qa;
  __ESBMC_loop_invariant(0 <= qa && qa < 256);
  __ESBMC_loop_invariant(qa % 2 == 1);
  for (idx = 0; idx < 10; ++idx)
  {
    qa = (qa + 2) % 256;
  }
  __ESBMC_assert(qa % 2 == 1, "post loop: qa mod");
  __ESBMC_assert(0 <= qa, "post loop: qa ge");
}
