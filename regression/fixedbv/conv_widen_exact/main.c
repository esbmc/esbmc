/* Widening fixed -> fixed conversions are exact, and widen-then-narrow
 * round-trips restore the value; pinned by native execution. */
#include <assert.h>

int main(void)
{
  short _Fract n = -0.6796875hr;
  assert((_Fract)n == -0.6796875r);
  assert((long _Fract)n == -0.6796875lr);
  assert((_Accum)n == -0.6796875k);
  assert((long _Accum)n == -0.6796875lk);

  unsigned short _Fract un = 0.75uhr;
  assert((unsigned _Fract)un == 0.75ur);
  assert((unsigned _Accum)un == 0.75uk);

  short _Accum ha = -3.5hk;
  assert((_Accum)ha == -3.5k);
  assert((long _Accum)ha == -3.5lk);

  assert((short _Fract)(long _Accum)n == n);
  assert((unsigned short _Fract)(unsigned _Accum)un == un);
  return 0;
}
