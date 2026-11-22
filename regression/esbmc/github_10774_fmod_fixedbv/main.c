#include <math.h>
int main()
{
  double r = fmod(5.0, 3.0);
  __ESBMC_assert(r == 2.0, "fmod(5,3) == 2");
  return 0;
}
