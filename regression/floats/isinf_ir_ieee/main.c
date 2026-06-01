// Verify isinf/isfinite/isnormal semantics under --ir-ieee integer encoding.
// Any finite float is bounded by FLT_MAX, so f > FLT_MAX implies isinf(f).
// Dually, isinf(f) implies !isfinite(f), and a normal float satisfies
// min_normal <= |f| <= max_normal.
#include <assert.h>
#include <math.h>

extern float __VERIFIER_nondet_float(void);

int main(void)
{
  float f = __VERIFIER_nondet_float();

  // f > FLT_MAX  =>  isinf(f) must hold
  if(f > 3.4028234663852886e+38f)
    assert(isinf(f));

  // isinf(f)  =>  !isfinite(f)
  if(isinf(f))
    assert(!isfinite(f));

  // A value known to be finite is not infinite
  float g = 1.0f;
  assert(!isinf(g));
  assert(isfinite(g));
  assert(isnormal(g));

  return 0;
}
