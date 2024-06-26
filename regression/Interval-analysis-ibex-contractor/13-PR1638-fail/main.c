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
  else if (x >= 20 || x < 0)
    x++;

  assert(x != y);

  return 0;
}