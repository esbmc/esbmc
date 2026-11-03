/* Issue #8701: whatever value fesetround rejects, the rounding direction is
 * left as it was (C11 7.6.3.2p2). */
#include <assert.h>
#include <fenv.h>
int nondet_int();
int main(void)
{
  assert(fesetround(FE_UPWARD) == 0);
  int mode = nondet_int();
  if (fesetround(mode) != 0)
    assert(fegetround() == FE_UPWARD);
  return 0;
}
