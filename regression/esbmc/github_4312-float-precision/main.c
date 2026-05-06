#include <assert.h>

// Regression for issue #4312-C: bit-distinct floats must render with
// round-trippable precision so users can tell witnesses apart. Before
// #4312, six witnesses on `assert(x > 0)` for a nondet float printed
// four near-FLT_MAX entries as the same `-1.701412e+38f` string at
// the default 6-digit precision. The fix passes presentationt::WITNESS
// at the witness print site, which enables UNIQUE_FLOAT_REPR (8 digits
// for binary32 — enough to round-trip).

extern float nondet_float(void);

int main(void)
{
  float x = nondet_float();
  assert(x > 0);
  return 0;
}
