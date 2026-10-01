#include <string.h>
#include <assert.h>

/* A loop bound copied in by memcpy. */
int main()
{
  int n = 3, m = nondet_int();
  memcpy(&n, &m, sizeof n);
  __ESBMC_assume(n >= 0 && n < 6);
  int c = 0;
  for (int i = 0; i < n; i++)
    c++;
  assert(n < 4);
  return 0;
}
