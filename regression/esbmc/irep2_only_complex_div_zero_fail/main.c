#include <complex.h>

int nondet_int(void);
int sink;

int main(void)
{
  /* The divisor check must land on the lowered denominator br*br + bi*bi,
     not on the complex operand: with both components nondet it can be 0.
     The operands are built through __real__/__imag__ rather than with `I`,
     which is float _Complex and would promote the division to ieee_div --
     exempt from the check. `sink` keeps the quotient live. */
  int complex a = 4, b = 0;
  __imag__ a = 2;
  __real__ b = nondet_int();
  __imag__ b = nondet_int();

  int complex w = a / b;
  sink = __real__ w;

  return 0;
}
