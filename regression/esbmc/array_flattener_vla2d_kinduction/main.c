#include <assert.h>

// Two-dimensional VLA under --array-flattener with k-induction. An ite between
// distinct unbounded arrays requires both to share one index set; replaying
// history in add_new_indexes before join_array_indexes broke that when a later
// select or ite added an index to only one side.
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
  assert(s >= 0);
  return 0;
}
