#include <math.h>
int main()
{
  int q;
  __ESBMC_assert(remainder(5.0, 3.0) == -1.0, "remainder(5,3) == -1");
  __ESBMC_assert(remquo(5.0, 3.0, &q) == -1.0 && q == 2, "remquo(5,3) == -1");
  return 0;
}
