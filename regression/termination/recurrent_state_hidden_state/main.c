// The first iteration changes only the rounding mode, which ESBMC keeps in an
// internal variable. The state that decides the loop's future includes it, so
// the second visit to the head is no repeat.
#include <fenv.h>

int main(void)
{
  int c = 0;
  while (1)
  {
    if (fegetround() == FE_TONEAREST)
    {
      fesetround(FE_DOWNWARD);
      continue;
    }
    c++;
    if (c > 3)
      break;
  }
  return 0;
}
