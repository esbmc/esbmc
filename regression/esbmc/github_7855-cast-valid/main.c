// esbmc/esbmc#7855 left the C-level integer round trip alone: a typecast means
// the numeric address, which identifies a pointer only for an object ESBMC
// knows about. This pins that case, which the bitcast change must not disturb.
#include <stdint.h>

int g;

int main(void)
{
  const void *key;
  __ESBMC_assume(key == &g || key == 0);
  uintptr_t address = (uintptr_t)key;
  const void *back = (const void *)address;

  __ESBMC_assert(back == key, "C17 7.20.1.4p1: the round trip compares equal");
  return 0;
}
