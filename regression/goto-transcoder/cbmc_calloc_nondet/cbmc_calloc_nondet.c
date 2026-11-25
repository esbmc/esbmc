#include <stdlib.h>
#include <assert.h>
unsigned nondet_uint(void);
int main()
{
  unsigned n = nondet_uint();
  if (n > 8)
    return 0;
  char *p = calloc(n, 1);
  if (p && n > 3)
    assert(p[3] == 0);
  return 0;
}
