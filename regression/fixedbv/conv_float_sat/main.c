/* Float -> _Sat fixed is defined for every input: out-of-range values and
 * infinities clamp to the rails, NaN converts to 0 (Clang's choice; the TR
 * leaves it undefined). Pinned by native execution. */
#include <assert.h>
#include <math.h>

int main(void)
{
  assert((_Sat _Fract)2.5f == 0.999969482421875r); /* s.15 max */
  assert((_Sat _Fract)INFINITY == 0.999969482421875r);
  assert((_Sat _Fract)NAN == 0.0r);
  assert((_Sat unsigned _Fract)-0.5f == 0.0ur);
  assert((_Sat short _Accum)1000.0f == 255.9921875hk);
  return 0;
}
