// esbmc/esbmc#7855: every store of a pointer into an untyped byte object goes
// through its numeric address, and that address does not reconstruct the same
// pointer when the pointer lies outside every object ESBMC knows about.
#include <stdint.h>

int main(void)
{
  const void *key;
  uintptr_t address = (uintptr_t)key;
  const void *back = (const void *)address;

  __ESBMC_assert(back == key, "C17 7.20.1.4p1: the round trip compares equal");
  return 0;
}
