/* #9371: C17 G.5.1p4 infinity properties of complex * and /. */
#include <assert.h>
#include <math.h>

int main(void)
{
  double _Complex inf_inf, one, zero, two_three, r;
  __real__ inf_inf = INFINITY;
  __imag__ inf_inf = INFINITY;
  __real__ one = 1.0;
  __imag__ one = 0.0;
  __real__ zero = 0.0;
  __imag__ zero = 0.0;
  __real__ two_three = 2.0;
  __imag__ two_three = 3.0;

  /* An infinity times a nonzero finite number is an infinity. */
  r = inf_inf * one;
  assert(isinf(__real__ r) || isinf(__imag__ r));
  r = one * inf_inf;
  assert(isinf(__real__ r) || isinf(__imag__ r));

  /* A nonzero finite number or an infinity over a zero is an infinity. */
  r = one / zero;
  assert(isinf(__real__ r) && __real__ r > 0);
  r = inf_inf / zero;
  assert(isinf(__real__ r) || isinf(__imag__ r));

  /* An infinity over a finite number is an infinity, a finite number over an
     infinity is a zero. */
  r = inf_inf / two_three;
  assert(isinf(__real__ r) || isinf(__imag__ r));
  r = two_three / inf_inf;
  assert(__real__ r == 0 && __imag__ r == 0);

  /* A product that overflows with a NaN part is an infinity (_Cmultd). */
  double _Complex big_nan, big;
  __real__ big_nan = 1e308;
  __imag__ big_nan = NAN;
  __real__ big = 1e308;
  __imag__ big = 0.0;
  r = big_nan * big;
  assert(isinf(__real__ r) || isinf(__imag__ r));

  /* Finite operands keep the usual formulas. */
  r = two_three * two_three;
  assert(__real__ r == -5.0 && __imag__ r == 12.0);
  r = two_three / one;
  assert(__real__ r == 2.0 && __imag__ r == 3.0);

  /* Through a compound assignment. */
  __real__ r = INFINITY;
  __imag__ r = NAN;
  r *= one;
  assert(isinf(__real__ r) || isinf(__imag__ r));
  return 0;
}
