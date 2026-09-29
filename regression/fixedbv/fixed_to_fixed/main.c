/* Fixed -> fixed conversions, pinned by native execution. */
#include <assert.h>

int main(void)
{
  _Accum k = -3.5k;
  _Accum wide = -3.5k;

  short _Fract narrowed = (short _Fract)(wide + 3.25k); /* -0.25 fits */
  assert(narrowed == -0.25hr);

  /* -3.5 clamps to the s.7 minimum -1.0 (raw 0x80); -1.0hr is -127/128,
   * see sat_clamp for the TR 4.1.5 literal rule. */
  assert((_Sat short _Fract)wide == -0.9921875hr - 0.0078125hr);

  assert((long _Accum)k == -3.5lk); /* widening is exact */
  return 0;
}
