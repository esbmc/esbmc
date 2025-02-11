#include <assert.h>

int main()
{
  int x;
  unsigned int y;
  __ESBMC_assume(x >= -2147483648 && x <= 2147483647);
  __ESBMC_assume(y >= 0 && y <= 4294967295);

  if (x < 1 && y < 1)
  {
    x = x + y;
  }
  else if (x >= 0)
    x++;

  __ESBMC_assume(y >= 2147483648 && y <= 4294967295);
  if (x > 0)
    assert(x < y);

  return 0;
}