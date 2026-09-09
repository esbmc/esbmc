/* A program's own sqrt over a non-float type must not be replaced by
 * ESBMC's ieee_sqrt intrinsic (esbmc/esbmc#6904 for the abs case). The
 * differential assertion is the sharp test: two byte-identical functions
 * must agree for every input, whatever they compute. */
#include <assert.h>

unsigned short _Fract sqrt(unsigned short _Fract x)
{
  return x >> 1; /* deliberately not a square root */
}

unsigned short _Fract mysqrt(unsigned short _Fract x)
{
  return x >> 1;
}

unsigned short _Fract nondet_ufract(void);

int main(void)
{
  unsigned short _Fract v = nondet_ufract();
  assert(sqrt(v) == mysqrt(v));
  assert(sqrt(0.5uhr) == 0.25uhr);
  return 0;
}
