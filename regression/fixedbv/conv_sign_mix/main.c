/* Signed <-> unsigned fixed conversions and _Sat rails at conversion
 * boundaries; pinned by native execution. The union pins the s.7 rails by
 * raw bit pattern (also exercising the fixed <-> raw-BV bridge). */
#include <assert.h>

int main(void)
{
  /* in-range sign mixes are value-preserving */
  short _Fract sp = 0.25hr;
  assert((unsigned short _Fract)sp == 0.25uhr);
  unsigned short _Fract up = 0.25uhr;
  assert((short _Fract)up == 0.25hr);

  /* u.8 above the s.7 max: non-Sat would be UB; _Sat clamps */
  unsigned short _Fract ubig = 0.99609375uhr; /* 255/256 */
  assert((_Sat short _Fract)ubig == 0.9921875hr);

  /* negative into _Sat unsigned clamps to zero */
  short _Fract sneg = -0.5hr;
  assert((_Sat unsigned short _Fract)sneg == 0.0uhr);
  _Accum kneg = -3.5k;
  assert((_Sat unsigned short _Accum)kneg == 0.0uhk);

  /* _Sat rails from a wider fixed source */
  _Accum kbig = 200.25k;
  assert((_Sat short _Accum)kbig == 200.25hk); /* fits */
  _Accum kbig2 = 300.5k;
  assert((_Sat short _Accum)kbig2 == 255.9921875hk);
  assert((_Sat short _Fract)kbig2 == 0.9921875hr);
  assert((_Sat unsigned short _Fract)kbig2 == 0.99609375uhr);

  /* raw bits of both s.7 rails */
  union
  {
    short _Fract f;
    signed char r;
  } pun;
  pun.f = (_Sat short _Fract)kneg; /* min */
  assert(pun.r == (signed char)0x80);
  pun.f = (_Sat short _Fract)kbig2; /* max */
  assert(pun.r == 0x7f);
  return 0;
}
