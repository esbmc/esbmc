#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void)
{
  __VERIFIER_assert(trunc(2.7) == 2.0);
  __VERIFIER_assert(trunc(-2.7) == -2.0);

  double c = trunc(-0.0);
  __VERIFIER_assert((c == -0.0) && signbit(c));

  c = trunc(-INFINITY);
  __VERIFIER_assert(isinf(INFINITY) && signbit(c));
}

