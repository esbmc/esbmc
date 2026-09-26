/* A user-defined abs() must not be replaced by ESBMC's builtin abs node
 * (esbmc/esbmc#6904). The builtin lowers to (x >= 0) ? x : -x, which
 * overflows at the format minimum; TR 18037's absfx saturates there
 * instead, so verifying the builtin reports a spurious failure -- and
 * would hide a real bug whenever the builtin happens to be correct.
 *
 * The differential assertion is the sharp test: two byte-identical
 * functions must agree for every input, whatever they compute. */
#include <assert.h>

#define FRACT_MIN (-0.9921875hr - 0.0078125hr)
#define FRACT_MAX 0.9921875hr

short _Fract abs(short _Fract x)
{
  if (x == FRACT_MIN)
    return FRACT_MAX;
  return (x < 0.0hr ? -x : x);
}

short _Fract myabs(short _Fract x)
{
  if (x == FRACT_MIN)
    return FRACT_MAX;
  return (x < 0.0hr ? -x : x);
}

short _Fract nondet_sfract(void);

int main(void)
{
  short _Fract v = nondet_sfract();
  assert(abs(v) == myabs(v));
  assert(abs(v) >= 0.0hr);
  return 0;
}
