#include <math.h>
int main()
{
  double r = fmod(5.0, 3.0);
  __ESBMC_assert(r == 1.0, "fmod(5,3) is 2, not 1");
  return 0;
}
