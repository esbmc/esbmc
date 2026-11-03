/* github #8682: atan2 special cases, C11 F.10.1.4 */
#include <assert.h>
#include <fenv.h>
#include <math.h>

double nondet_double(void);

int main(void)
{
  assert(atan2(-0.0, -1.0) == -M_PI);
  assert(atan2(0.0, -0.0) == M_PI);
  assert(atan2(-0.0, -0.0) == -M_PI);
  assert(atan2(0.0, 0.0) == 0.0 && !signbit(atan2(0.0, 0.0)));
  assert(atan2(-0.0, 1.0) == 0.0 && signbit(atan2(-0.0, 1.0)));
  assert(isnan(atan2(NAN, 0.0)));
  assert(isnan(atan2(1.0, NAN)));
  assert(atan2(INFINITY, INFINITY) == M_PI_4);
  assert(atan2(-INFINITY, -INFINITY) == -3 * M_PI_4);
  assert(atan2(-INFINITY, 1.0) == -M_PI_2);
  assert(atan2(1.0, -INFINITY) == M_PI);
  assert(atan2(1.0, 1.0) > 0.78 && atan2(1.0, 1.0) < 0.79);

  fesetround(FE_DOWNWARD);
  assert(acos(-1.0) == M_PI);
  assert(acos(1.0) == 0.0 && !signbit(acos(1.0)));
  assert(asin(-0.0) == 0.0 && signbit(asin(-0.0)));
  fesetround(FE_TONEAREST);

  double u = nondet_double();
  __ESBMC_assume(u < 0.0);
  assert(atan2(-0.0, u) == -M_PI);
  return 0;
}
