#include <assert.h>

int nondet_int(void);

int main(void)
{
  /* An integral element type is the arm that stays on plain add/sub/mul/div,
     where irep2_only_complex_arith covers the ieee_* one. Each expected value
     is a scalar expression over the components, so a wrong per-component
     formula is not restated on both sides of the comparison. */
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
