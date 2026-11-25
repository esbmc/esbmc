#include <assert.h>
#include <stdlib.h>

unsigned long nondet_ulong(void);

int main()
{
  unsigned long n = nondet_ulong();
  unsigned long k = nondet_ulong();
  char *p = calloc(n, 4);
  if (p && k < 4 * n)
    assert(p[k] == 1);
  free(p);
  return 0;
}
