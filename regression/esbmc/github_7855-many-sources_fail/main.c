// esbmc/esbmc#7855: a slot other than the one the key went into need not
// hold it.
#include <stdlib.h>

const void *nondet_ptr(void);

int main(void)
{
  unsigned long n = nondet_ulong();
  __ESBMC_assume(n >= 42 && n <= 44);
  const void **slots = malloc(n * sizeof(void *));
  if (!slots)
    return 0;

  for (int k = 0; k < 40; ++k)
    slots[k] = nondet_ptr();

  const void *key;
  unsigned long i = nondet_ulong();
  __ESBMC_assume(i >= 40 && i < n);
  slots[i] = key;

  __ESBMC_assert(slots[0] == key, "another slot holds the key");
  return 0;
}
