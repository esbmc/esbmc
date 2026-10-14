/* Test that __ESBMC_unroll(N) binds to a while loop that declares a variable
 * in its condition (`while (int v = e)`). The loop preamble holds the DECL
 * (and the side-effecting computation) of the condition, all of which must be
 * skipped when binding the intrinsic to the loop. This used to silently fail
 * to match, leaving the loop unbounded.
 *
 * Note: the C++ frontend currently hoists the condition's computation out of
 * such loops, so unwinding assertions are disabled here; the point of the
 * test is that the intrinsic is applied to the right loop (see the expected
 * "#pragma unroll" line).
 */

int main()
{
  int sum = 0;
  int n = 5;

  __ESBMC_unroll(5);
  while(int v = n--)
    sum += v;

  return 0;
}
