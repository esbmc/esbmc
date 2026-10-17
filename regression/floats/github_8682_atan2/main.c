// #8682: atan2 must follow C11 F.10.1.4 for signed zeros, NaN and infinities.
#include <assert.h>
#include <math.h>
int main(void)
{
  assert(atan2(-0.0, -1.0) == -M_PI);
  assert(atan2(0.0, -0.0) == M_PI);
  assert(atan2(-0.0, -0.0) == -M_PI);
  assert(signbit(atan2(-0.0, 1.0)) && atan2(-0.0, 1.0) == 0.0);
  assert(!signbit(atan2(0.0, 0.0)));
  assert(signbit(atan2(-0.0, 0.0)));
  assert(isnan(atan2(NAN, 0.0)));
  assert(isnan(atan2(1.0, NAN)));
  assert(atan2(INFINITY, INFINITY) == M_PI_4);
  assert(atan2(-INFINITY, INFINITY) == -M_PI_4);
  assert(atan2(INFINITY, -INFINITY) == 3 * M_PI_4);
  assert(atan2(-INFINITY, -INFINITY) == -3 * M_PI_4);
  assert(atan2(INFINITY, 2.0) == M_PI_2);
  assert(atan2(-INFINITY, -2.0) == -M_PI_2);
  assert(atan2(1.0, INFINITY) == 0.0 && !signbit(atan2(1.0, INFINITY)));
  assert(signbit(atan2(-1.0, INFINITY)));
  assert(atan2(1.0, -INFINITY) == M_PI);
  assert(atan2(-1.0, -INFINITY) == -M_PI);
  assert(atan2(1.0, 0.0) == M_PI_2);
  assert(atan2(-1.0, -0.0) == -M_PI_2);
  assert(atan2(1.0, 1.0) > 0.785 && atan2(1.0, 1.0) < 0.786);
  assert(atan2(-1.0, -1.0) < -2.356 && atan2(-1.0, -1.0) > -2.357);
  return 0;
}
