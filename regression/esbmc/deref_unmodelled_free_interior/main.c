#include <stdint.h>
#include <stdlib.h>

unsigned long nondet_ulong(void);
char a[64];

/* Freeing an interior heap pointer the value set cannot resolve. */
int main(void)
{
  char *q = malloc(32);
  if (!q)
    return 0;
  __ESBMC_assume((uintptr_t)q != 0);
  unsigned long u = nondet_ulong();
  char *p = (char *)(u * 8 - (uintptr_t)&a[0]);
  if ((uintptr_t)p == (uintptr_t)q + 8)
    free(p);
  return 0;
}
