#include <fenv.h>
int main()
{
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= 20 && fegetround() == FE_TONEAREST);
  while (i < 20)
    i++;
  __ESBMC_assert(fegetround() == FE_TONEAREST, "rounding mode kept");
  return 0;
}
