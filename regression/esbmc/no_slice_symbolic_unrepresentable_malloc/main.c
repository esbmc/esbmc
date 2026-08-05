#include <stdlib.h>
#include <assert.h>

size_t nondet_size(void);

int main()
{
  // Same defect reached through a symbolic size: the address-space model
  // cannot lay the object out, so the path is vacuously infeasible and the
  // reachable assert(0) is missed.
  size_t n = nondet_size();
  __ESBMC_assume(n >= 0xFFFFFFFFFFFFFFF0UL);
  void *b = malloc(n);
  assert(0);
}
