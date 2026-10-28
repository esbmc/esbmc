// esbmc/esbmc#7855: a pointer read twice from bytes nothing stored a pointer
// into is one pointer, whatever is allocated between the reads.
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
  const void *second = slots[i];

  __ESBMC_assert(first == second, "two reads of one slot agree");
  return 0;
}
