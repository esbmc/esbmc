#include <assert.h>
#include <float.h>

// Regression for issue #4312-C: bit-distinct floats must render with
// round-trippable precision so users can tell witnesses apart. Before
// #4312, six witnesses on `assert(x > 0)` for a nondet float printed
// four near-FLT_MAX entries as the same `-1.701412e+38f` string at
// the default 6-digit precision. The fix passes presentationt::WITNESS
// at the witness print site, which enables UNIQUE_FLOAT_REPR (8 digits
// for binary32 — enough to round-trip).
//
// Pin x to the finite near-(-FLT_MAX) regime where the original bug
// manifested. Without the assume the solver is also free to pick NaN,
// -Infinity, or near-zero denormals — all valid violating inputs but
// not what this regression is here to exercise. NaN comparisons are
// false so `x >= -FLT_MAX` filters NaN as well as -Infinity.

extern float nondet_float(void);

int main(void)
{
  float x = nondet_float();
  __ESBMC_assume(x < -1.0e+38f && x >= -FLT_MAX);
  assert(x > 0);
  return 0;
}
