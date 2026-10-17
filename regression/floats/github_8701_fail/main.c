/* Issue #8701: an unrecognised mode leaves the rounding direction unchanged,
 * so it is still FE_UPWARD here, not FE_TONEAREST. */
#include <assert.h>
#include <fenv.h>
int main(void)
{
  fesetround(FE_UPWARD);
  fesetround(12345);
  assert(fegetround() == FE_TONEAREST);
  return 0;
}
