#include <assert.h>

int main(void)
{
  _Bool b = 1;
  __ESBMC_assume(b != 0);
  b ^= 1;
  assert(b == 1);
  return 0;
}
