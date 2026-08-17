#include <assert.h>

int f(void)
{
  __ESBMC_ensures(__ESBMC_return_value == 1);
  return 1;
}

int main(void)
{
  int r = f();
  assert(r == 1);
  return 0;
}
