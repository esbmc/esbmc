#include <stdint.h>
int g[2];
int main(void)
{
  uintptr_t u = 0;
  while (1)
  {
    __ESBMC_assert((int *)u != &g[1], "u never names g[1]");
    u = u + sizeof(int);
  }
  return 0;
}
