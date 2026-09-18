// esbmc/esbmc#7855: the github_7855-cast round trip already holds once the
// pointer is pinned to an object ESBMC knows about.
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
