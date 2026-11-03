/* Issue #8701: fesetround of an unrecognised mode must fail and leave the
 * rounding direction unchanged (C11 7.6.3.2). */
#include <assert.h>
#include <fenv.h>
int main(void)
{
  assert(fesetround(FE_UPWARD) == 0);
  assert(fesetround(12345) != 0);
  assert(fegetround() == FE_UPWARD);
  assert(fesetround(FE_TONEAREST) == 0);
  assert(fegetround() == FE_TONEAREST);
  return 0;
}
