// Exercises the CBMC->ESBMC fpclassify bridge: a CBMC <math.h> lowers
// fpclassify(x) to a call to __fpclassify{f,d,l}(x), which the goto binary
// leaves bodyless. ESBMC must link its own operational-model body so the
// classification is computed rather than left nondet. We call the internal
// symbols directly (exactly what fpclassify expands to) so the bridge is
// exercised on any host -- glibc's fpclassify macro would otherwise fold to
// __builtin_fpclassify and never reach the bridge. The FP_* codes come from
// this host's <math.h>, matching the values ESBMC's model returns.
#include <assert.h>
#include <math.h>
extern int __fpclassifyf(float);
extern int __fpclassifyd(double);
extern int __fpclassifyl(long double);
int main()
{
  double n = 1.0, z = 0.0, inf = 1.0 / 0.0, nan = 0.0 / 0.0;
  double sub = 0x1p-1030; // subnormal double
  assert(__fpclassifyd(n) == FP_NORMAL);
  assert(__fpclassifyd(z) == FP_ZERO);
  assert(__fpclassifyd(inf) == FP_INFINITE);
  assert(__fpclassifyd(nan) == FP_NAN);
  assert(__fpclassifyd(sub) == FP_SUBNORMAL);
  assert(__fpclassifyf(2.0f) == FP_NORMAL);
  assert(__fpclassifyl(3.0L) == FP_NORMAL);
  return 0;
}
