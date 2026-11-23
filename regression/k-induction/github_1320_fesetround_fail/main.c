/* fesetround writes a model variable the havoc does not cover, so the
   inductive step must still follow the paths the bound cuts out of the loop
   (esbmc/esbmc#1320). */
#include <fenv.h>
int main()
{
  unsigned i = 0;
  while (i < 20)
  {
    if (i == 0)
      fesetround(FE_UPWARD);
    i++;
  }
  __ESBMC_assert(fegetround() == FE_TONEAREST, "rounding mode restored");
  return 0;
}
