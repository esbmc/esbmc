#include <assert.h>
#include <complex.h>

int nondet_int(void);

int main(void)
{
  /* (4+2i) / (1+1i) = 3 - 1i, (1+2i)*(3+4i) - (1+1i) = -6 + 9i */
  double complex z = (4.0 + 2.0 * I) / (1.0 + 1.0 * I);
  assert(creal(z) == 3.0 && cimag(z) == -1.0);
  double complex y = (1.0 + 2.0 * I) * (3.0 + 4.0 * I) - (1.0 + 1.0 * I);
  assert(creal(y) == -6.0 && cimag(y) == 9.0);

  /* `I` is float _Complex, so anything built with it promotes to a floating
     element type and the components land on ieee_*. Building the operands
     through __real__/__imag__ keeps the element type integral, which is the
     arm that stays on plain add/sub/mul/div. Each expected value is a scalar
     expression over the components, so a wrong per-component formula is not
     restated on both sides of the comparison. */
  int complex a = 4, b = 0;
  __imag__ a = 2;
  __real__ b = nondet_int();
  __imag__ b = nondet_int();
  int br = __real__ b, bi = __imag__ b;
  __ESBMC_assume(br >= 1 && br <= 4 && bi >= 1 && bi <= 4);

  int complex s = a + b;
  assert(__real__ s == 4 + br && __imag__ s == 2 + bi);
  int complex d = a - b;
  assert(__real__ d == 4 - br && __imag__ d == 2 - bi);
  int complex m = a * b;
  assert(__real__ m == 4 * br - 2 * bi && __imag__ m == 4 * bi + 2 * br);
  int complex q = a / b;
  assert(__real__ q == (4 * br + 2 * bi) / (br * br + bi * bi));
  assert(__imag__ q == (2 * br - 4 * bi) / (br * br + bi * bi));

  return 0;
}
