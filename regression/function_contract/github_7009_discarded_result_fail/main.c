/* A dropped result still needs naming. The ensures is assumed either way, and
 * with nothing to rewrite __ESBMC_return_value to it was assumed over a symbol
 * no instruction defines, which took the reachable assertion below with it. */
#include <assert.h>

int g(void)
{
  __ESBMC_ensures(__ESBMC_return_value == 1);
  return 1;
}

int main(void)
{
  g();
  assert(0);
  return 0;
}
