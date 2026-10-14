#include <assert.h>
int main()
{
  float n = 0.0f / 0.0f; /* NaN */
  /* IEEE-754: NaN != NaN is true, so this assertion must hold. A bitwise
     equality would make it false -- this pins the IEEE semantics of the
     ieee_float_notequal -> notequal rewrite. */
  assert(n != n);
  return 0;
}
