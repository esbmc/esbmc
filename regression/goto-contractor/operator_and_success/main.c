#include <assert.h>
int main()
{
  int x, y;
  x = 1;
  y = 0;
  __ESBMC_assume(x >= 0);
  __ESBMC_assume(y >= 0);
  while(y <= 1000 && nondet_int())
  {
    y++;
    x += y;
  }
  assert(x >= y && x >=-y);
  return 0;
}