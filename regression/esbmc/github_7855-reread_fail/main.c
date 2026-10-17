// esbmc/esbmc#7855: reads of two different uninitialised slots are not forced
// to agree.
#include <stdlib.h>

int main(void)
{
  const void **slots = malloc(2 * sizeof(void *));
  if (!slots)
    return 0;

  unsigned long i = nondet_ulong();
  __ESBMC_assume(i < 2);

  const void **slot = &slots[i];
  const void *first = *slot;
  void *fresh = malloc(1);
  const void *second = slots[1 - i];

  __ESBMC_assert(first == second, "two different slots agree");
  return 0;
}
