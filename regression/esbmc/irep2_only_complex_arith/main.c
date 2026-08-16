#include <assert.h>

int main(void)
{
  /* `_Complex` and __real__/__imag__ rather than <complex.h>'s `complex`, `I`
     and creal/cimag: MSVC ships no C99 complex header, so the macro spellings
     do not parse there.

     The integral element type is exercised by irep2_only_complex_arith_int.
     Either arm alone solves in milliseconds; both in one formula take z3 past
     20 minutes, which timed the Windows job out. */

  /* (4+2i) / (1+1i) = 3 - 1i, (1+2i)*(3+4i) - (1+1i) = -6 + 9i. A double
     element type puts the components on ieee_*. */
  double _Complex fa, fb, fc;
  __real__ fa = 4.0;
  __imag__ fa = 2.0;
  __real__ fb = 1.0;
  __imag__ fb = 1.0;
  double _Complex z = fa / fb;
  assert(__real__ z == 3.0 && __imag__ z == -1.0);

  __real__ fa = 1.0;
  __imag__ fa = 2.0;
  __real__ fb = 3.0;
  __imag__ fb = 4.0;
  __real__ fc = 1.0;
  __imag__ fc = 1.0;
  double _Complex y = fa * fb - fc;
  assert(__real__ y == -6.0 && __imag__ y == 9.0);

  return 0;
}
