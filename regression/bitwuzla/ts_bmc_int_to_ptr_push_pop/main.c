#include <stdint.h>
int g[2];
int main(void)
{
  for (uintptr_t u = 0; u < 16; u += sizeof(int))
    __ESBMC_assert(
      (int *)u != &g[1] || u == (uintptr_t)&g[1],
      "only g[1]'s address names g[1]");
  return 0;
}
