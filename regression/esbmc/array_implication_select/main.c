#include <assert.h>

// Symbolic read from a bounded array under --array-implication-select, which
// encodes the select as implications on a free variable instead of a nested
// ite chain (esbmc/esbmc#82). The assertion pins which element each index
// selects, not merely that the result came from the array: a guard or index
// off-by-one in the encoding must be refutable here.
int main()
{
  int a[8] = {10, 20, 30, 40, 50, 60, 70, 80};

  unsigned i;
  __ESBMC_assume(i < 8);

  int x = a[i];
  assert(x == 10 * (int)(i + 1));
  return 0;
}
