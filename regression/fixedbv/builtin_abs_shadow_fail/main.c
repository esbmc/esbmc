/* The user's abs is genuinely wrong here (no saturation at the minimum),
 * so it must be refuted. Guards the opposite failure mode of
 * builtin_abs_shadow: if ESBMC substituted a *correct* builtin for the
 * program's code, this bug would be hidden. */
#include <assert.h>

short _Fract abs(short _Fract x)
{
  return (x < 0.0hr ? -x : x); /* overflows at the format minimum */
}

short _Fract nondet_sfract(void);

int main(void)
{
  short _Fract v = nondet_sfract();
  assert(abs(v) >= 0.0hr);
  return 0;
}
