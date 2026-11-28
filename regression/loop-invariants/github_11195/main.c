#include <fenv.h>
int main()
{
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= 20);
  while (i < 20)
  {
    if (i == 0)
      fesetround(FE_UPWARD);
    i++;
  }
  __ESBMC_assert(fegetround() == FE_TONEAREST, "rounding mode restored");
  return 0;
}
