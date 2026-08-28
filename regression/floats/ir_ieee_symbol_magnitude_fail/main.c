// Anti-vacuity twin: the preconditions above must leave f free, so a false
// claim about it still fails.
#include <assert.h>

#define INF (1.0 / 0.0)

int main(void)
{
  double f;
  __ESBMC_assume(f == f);
  __ESBMC_assume(f != INF && f != -INF);

  assert(0 * f == 1);
  return 0;
}
