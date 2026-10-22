#include <assert.h>

// Failing counterpart of array_flattener_vla2d_kinduction: the same 2D VLA
// under --array-flattener with k-induction, where the sum is 1 once n is 2.
short nondet_short(void);

int main(void)
{
  long long n = nondet_short();
  __ESBMC_assume(n > 0 && n < 4);
  int a[n][n];
  for (int r = 0; r < n; r++)
    for (int c = 0; c < n; c++)
      a[r][c] = r > c ? 1 : 0;
  int s = 0;
  for (int r = 0; r < n; r++)
    for (int c = 0; c < n; c++)
      s += a[r][c];
  assert(s == 0);
  return 0;
}
