int nondet_int(void);
int sink;

int main(void)
{
  /* The divisor check must land on the lowered denominator br*br + bi*bi,
     not on the complex operand: with both components nondet it can be 0.
     The element type is kept integral so the division stays on div rather
     than ieee_div, which the check exempts. `sink` keeps the quotient live.
     _Complex rather than <complex.h>'s `complex`: MSVC ships no C99 header. */
  int _Complex a = 4, b = 0;
  __imag__ a = 2;
  __real__ b = nondet_int();
  __imag__ b = nondet_int();

  int _Complex w = a / b;
  sink = __real__ w;

  return 0;
}
