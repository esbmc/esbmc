/* Regression test for GitHub issue #4052 (variant with assert inside loop).
 *
 * The in-loop assertion "in loop: false" fails, but post-loop assertions
 * should still be individually proven and reported as PASSED by the
 * inductive step.  Before the fix, post-loop assertions were silently
 * swallowed when --multi-property was combined with --loop-invariant.
 *
 * Expected: VERIFICATION FAILED (due to "in loop: false"),
 * with post-loop assertions individually shown as PASSED.
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
    if (qa % 5 == 1)
    {
      __ESBMC_assert(0, "in loop: false");
      break;
    }
    qa = (qa + 2) % 256;
  }
  __ESBMC_assert(qa % 2 == 1, "post loop: qa mod");
  __ESBMC_assert(0 <= qa, "post loop: qa ge");
}
