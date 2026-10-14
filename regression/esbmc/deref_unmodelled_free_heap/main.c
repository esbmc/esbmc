#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

unsigned long nondet_ulong(void);
char a[64];

/* A valid free through an unresolved pointer; the assert shows it is reached. */
int main(void)
{
  char *q = malloc(16);
  if (!q)
    return 0;
  __ESBMC_assume((uintptr_t)q != 0);
  unsigned long u = nondet_ulong();
  char *p = (char *)(u * 8 - (uintptr_t)&a[0]);
  if ((uintptr_t)p == (uintptr_t)q)
  {
    free(p);
    __ESBMC_assert(0, "valid free is reachable");
  }
  return 0;
}
