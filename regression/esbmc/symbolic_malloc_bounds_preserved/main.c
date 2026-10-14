#include <stdlib.h>

size_t nondet_size(void);

int main()
{
  // Bounding a symbolic request to the address space must not weaken the
  // bounds check on a request that fits: b[n] is still out of range.
  size_t n = nondet_size();
  __ESBMC_assume(n >= 10 && n <= 100);
  char *b = malloc(n);
  if (b)
    b[n] = 1;
  return 0;
}
