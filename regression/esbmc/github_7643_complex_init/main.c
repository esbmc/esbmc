// A two-element brace initialiser gives a complex object its real and
// imaginary parts, a clang extension (github #7643).
#include <assert.h>

int nondet_int(void);

_Complex int gi = {1, 2};

int main(void)
{
  _Complex float fl = {1.5, 2};
  assert(__real__ gi == 1 && __imag__ gi == 2);
  assert(__real__ fl == 1.5f && __imag__ fl == 2.0f);

  int re = nondet_int(), im = nondet_int();
  _Complex int z = {re, im};
  assert(__real__ z == re && __imag__ z == im);
  return 0;
}
