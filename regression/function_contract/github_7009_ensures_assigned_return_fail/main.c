#include <assert.h>

int f(void)
{
  __ESBMC_ensures(__ESBMC_return_value == 1);
  return 1;
}

int main(void)
{
  int r = 200;
  r = f();
  assert(0);
  return 0;
}
