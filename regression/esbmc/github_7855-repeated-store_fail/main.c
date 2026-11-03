// esbmc/esbmc#7855: the slot stored over and over need not hold the key.
#include <stdlib.h>

int main(void)
{
  unsigned long n = nondet_ulong();
  __ESBMC_assume(n >= 2 && n <= 3);
  const void **slots = malloc(n * sizeof(void *));
  if (!slots)
    return 0;

  unsigned long i = nondet_ulong(), j = nondet_ulong();
  __ESBMC_assume(i < n && j < n && i != j);
  const void *key, *other;
  slots[i] = key;
  for (int k = 0; k < 40; ++k)
    slots[j] = other;

  __ESBMC_assert(slots[j] == key, "the other slot holds the key");
  return 0;
}
