#include <assert.h>

// Failing counterpart to array_implication_select: shifting the expected
// element by one must be refutable, so the flat select encoding cannot be
// satisfied by an off-by-one index mapping (esbmc/esbmc#82).
int main()
{
  int a[8] = {10, 20, 30, 40, 50, 60, 70, 80};

  unsigned i;
  __ESBMC_assume(i < 8);

  int x = a[i];
  assert(x == 10 * (int)(i + 2));
  return 0;
}
