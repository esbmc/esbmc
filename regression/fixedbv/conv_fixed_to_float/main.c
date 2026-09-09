/* Fixed -> float/double rounds to nearest even and is always defined (every
 * fixed-point range fits inside any float range). Values pinned by native
 * execution. */
#include <assert.h>

int main(void)
{
  _Accum k = 1.5k;
  assert((float)k == 1.5f);
  assert((double)k == 1.5);

  short _Fract n = -0.6796875hr; /* -87/128, exact in binary */
  assert((float)n == -0.6796875f);
  assert((double)n == -0.6796875);

  unsigned short _Fract u = 0.75uhr;
  assert((float)u == 0.75f);

  /* s.31 raw 0x40000040 is an exact tie for float's 24-bit significand:
   * round-to-nearest-EVEN keeps the even candidate (0.5), where
   * round-to-nearest-away or a second rounding step would not. */
  long _Fract tie = 0.5lr + 0.00000002980232238769531250lr;
  assert((float)tie == 0.5f);

  _Accum rt = 2.25k;
  assert((_Accum)(float)rt == rt); /* exact round trip */
  return 0;
}
