// vmaxq_f32/vminq_f32 and vmaxq_f64 on nondet lanes follow FMAX/FMIN: a NaN
// lane gives NaN, equal lanes give that value, max(-0, +0) is +0 and
// min(-0, +0) is -0 in any rounding mode, otherwise the larger or smaller.
#include <arm_neon.h>
#include <assert.h>
#include <fenv.h>
#include <math.h>

float nondet_float(void);
double nondet_double(void);
int nondet_int(void);

int main(void)
{
  float x = nondet_float(), y = nondet_float();
  float32x4_t a = {x, x, x, -0.0f}, b = {y, x, -x, 0.0f};
  int mode = nondet_int();
  __ESBMC_assume(
    mode == FE_TONEAREST || mode == FE_DOWNWARD || mode == FE_UPWARD ||
    mode == FE_TOWARDZERO);
  fesetround(mode);
  float32x4_t mx = vmaxq_f32(a, b), mn = vminq_f32(a, b);

  if (isnan(x) || isnan(y))
    assert(isnan(mx[0]) && isnan(mn[0]));
  else if (x != y)
    assert(mx[0] == (x > y ? x : y) && mn[0] == (x < y ? x : y));
  if (!isnan(x))
  {
    assert(mx[1] == x && mn[1] == x);
    assert(signbit(mx[1]) == signbit(x) && signbit(mn[1]) == signbit(x));
  }
  if (x == 0)
  {
    assert(mx[2] == 0 && !signbit(mx[2]));
    assert(mn[2] == 0 && signbit(mn[2]));
  }
  assert(!signbit(mx[3]) && signbit(mn[3]));

  double d = nondet_double();
  __ESBMC_assume(!isnan(d));
  float64x2_t p = {d, -0.0}, q = {d, 0.0};
  float64x2_t dx = vmaxq_f64(p, q), dn = vminq_f64(p, q);
  assert(dx[0] == d && dn[0] == d);
  assert(!signbit(dx[1]) && signbit(dn[1]));
  return 0;
}
