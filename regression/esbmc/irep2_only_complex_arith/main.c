#include <assert.h>

int nondet_int(void);

int main(void)
{
  /* `_Complex` and __real__/__imag__ rather than <complex.h>'s `complex`, `I`
     and creal/cimag: MSVC ships no C99 complex header, so the macro spellings
     do not parse there. */

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

  /* An integral element type is the arm that stays on plain add/sub/mul/div.
     Each expected value is a scalar expression over the components, so a wrong
     per-component formula is not restated on both sides of the comparison. */
  int _Complex a = 4, b = 0;
  __imag__ a = 2;
  __real__ b = nondet_int();
  __imag__ b = nondet_int();
  int br = __real__ b, bi = __imag__ b;
  __ESBMC_assume(br >= 1 && br <= 4 && bi >= 1 && bi <= 4);

  int _Complex s = a + b;
  assert(__real__ s == 4 + br && __imag__ s == 2 + bi);
  int _Complex d = a - b;
  assert(__real__ d == 4 - br && __imag__ d == 2 - bi);
  int _Complex m = a * b;
  assert(__real__ m == 4 * br - 2 * bi && __imag__ m == 4 * bi + 2 * br);
  int _Complex q = a / b;
  assert(__real__ q == (4 * br + 2 * bi) / (br * br + bi * bi));
  assert(__imag__ q == (2 * br - 4 * bi) / (br * br + bi * bi));

  return 0;
}
