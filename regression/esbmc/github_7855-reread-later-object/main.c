// esbmc/esbmc#7855: bytes read before an allocation may hold the address it
// later returns, and then name that object on every read.
#include <stdint.h>
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

  if (fresh && first == second && (uintptr_t)first == (uintptr_t)fresh)
    __ESBMC_assert(0, "a stale read names the later object");
  return 0;
}
