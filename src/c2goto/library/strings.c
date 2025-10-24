#include <strings.h>

int ffs(int x)
{
__ESBMC_HIDE:;
  if (x == 0)
    return 0;

  int pos = 1;
  while ((x & 1) == 0)
  {
    x >>= 1;
    pos++;
  }

  return pos;
}
