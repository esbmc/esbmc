// esbmc/esbmc#7855: once more pointers than the rebuild cap reach a read, the
// read back must still keep the latest ones rather than lose them all.
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

  __ESBMC_assert(slots[i] == key, "the stored key reads back");
  return 0;
}
