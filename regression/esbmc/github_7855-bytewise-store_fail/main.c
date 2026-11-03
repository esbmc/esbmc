// esbmc/esbmc#7855: a slot other than the one the key went into need not
// hold it.
#include <stdlib.h>

int main(void)
{
  unsigned long n = nondet_ulong();
  __ESBMC_assume(n >= 9 && n <= 10);
  const void **slots = malloc(n * sizeof(void *));
  if (!slots)
    return 0;

  const void *p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7;
  slots[0] = p0;
  slots[1] = p1;
  slots[2] = p2;
  slots[3] = p3;
  slots[4] = p4;
  slots[5] = p5;
  slots[6] = p6;
  slots[7] = p7;

  const void *key;
  unsigned long i = nondet_ulong();
  __ESBMC_assume(i >= 8 && i < n);
  slots[i] = key;

  __ESBMC_assert(slots[0] == key, "another slot holds the key");
  return 0;
}
