#include <assert.h>

int main()
{
  int x = 5;
  __ESBMC_init_object(&x);
  assert(x == 5);
  return 0;
}
