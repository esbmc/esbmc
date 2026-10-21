#include <assert.h>

int main()
{
  int x = 0;
  for (int i = 0; i < 3; i++)
  {
    x++;
    __ESBMC_ltl_sample();
  }
  assert(x == 3);
  return 0;
}
