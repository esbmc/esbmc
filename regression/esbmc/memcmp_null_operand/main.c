#include <assert.h>
#include <string.h>

unsigned nondet_uint();

int main()
{
  char a[4] = {1, 2, 3, 4}, b[4] = {1, 2, 3, 5};
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 4);
  // A null operand is fine when no byte is read (C2y 7.26.1, N3322).
  char *p = n ? a : 0;
  int r = memcmp(b, p, n);
  assert(n < 4 ? r == 0 : r > 0);
  return 0;
}
