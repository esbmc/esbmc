#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void)
{

  __VERIFIER_assert(remainder(5.1f, 3) == -0x1.ccccdp-1);
  __VERIFIER_assert(remainder(5.1f, -3) == -0x1.ccccdp-1);
  __VERIFIER_assert(remainder(-5.1f, -3) == 0x1.ccccdp-1);
  __VERIFIER_assert(remainder(-5.1f, 3) == 0x1.ccccdp-1);

  double rem = remainder(0.0, 1);
  __VERIFIER_assert((rem == 0.0) && (!signbit(rem)));

  rem = remainder(-.0, 1);
  __VERIFIER_assert((rem == 0.0) && (signbit(rem)));

  __VERIFIER_assert(isnan(remainder(5.1, 0)));
  __VERIFIER_assert(remainder(5.1, INFINITY) == 5.1);
}
