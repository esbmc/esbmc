/* --multi-property --loop-invariant with a CORRECT invariant.
 *
 * The invariant "res == (idx*(idx+1))/2" precisely captures the partial sum,
 * so both the inductive step and the post-loop assertion should be proven.
 * The final verdict must be VERIFICATION SUCCESSFUL.
 * Regression test for GitHub issue #3818.
 */
#include <assert.h>

unsigned int sum_loop(unsigned int n)
{
  unsigned int res = 0;
  unsigned int idx = 0;
  __ESBMC_assume(n < 10);
  __ESBMC_loop_invariant(n < 10);
  __ESBMC_loop_invariant(res == (idx * (idx + 1)) / 2); /* correct */
  __ESBMC_loop_invariant(idx <= n);
  while(idx < n)
  {
    ++idx;
    res += idx;
  }
  return res;
}

int main(void)
{
  unsigned int n;
  __ESBMC_assume(n < 10);
  unsigned int r = sum_loop(n);
  assert(r == (n * (n + 1)) / 2);
  return 0;
}
