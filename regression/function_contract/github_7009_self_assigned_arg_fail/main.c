/* The result overwrites the variable the ensures was instantiated with, so a
 * havoc of that place in situ makes the clause read `r == r + 1` and the path
 * dies. The caller then proves anything, including this. */
#include <assert.h>

int f(int x)
{
  __ESBMC_requires(x > 0);
  __ESBMC_ensures(__ESBMC_return_value == x + 1);
  return x + 1;
}

int main(void)
{
  int r = 5;
  r = f(r);
  assert(r == 5);
  return 0;
}
