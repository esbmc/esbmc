#include <assert.h>
#include <stdlib.h>

/* Two live local arrays never share an address. The malloc/free brings
 * __ESBMC_alloc into the formula, which only tracks heap objects. */
int main(void)
{
  char *m = malloc(1);
  char a[16];
  char b[16];
  if (m)
    free(m);
  assert((unsigned long)a != (unsigned long)b);
  return 0;
}
