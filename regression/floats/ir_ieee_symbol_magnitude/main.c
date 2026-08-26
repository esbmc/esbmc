// A float symbol's real value must be representable: at most max_normal in
// magnitude, or exactly the infinity sentinel. Left unconstrained it could
// land between the two, where |f| > max_normal reads as infinite while
// f != INFINITY reads as finite, and IEEE_MUL's invalid-operation term then
// gave 0*f a NaN predicate.
#include <assert.h>
#include <float.h>

#define INF (1.0 / 0.0)

int main(void)
{
  double f;
  __ESBMC_assume(f == f);
  __ESBMC_assume(f != INF && f != -INF);

  assert(f <= DBL_MAX);
  assert(f >= -DBL_MAX);
  assert(0 * f == 0);
  assert(f * 0 == 0);
  assert(f - f == 0);
  return 0;
}
