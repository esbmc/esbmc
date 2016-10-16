#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void)
{
  __VERIFIER_assert(fabs(+3.0) == +3.0);
  __VERIFIER_assert(fabs(-3.0) == +3.0);
  __VERIFIER_assert(fabs(-0.0) == -0.0);
  __VERIFIER_assert(fabs(-0.0) == +0.0);
  __VERIFIER_assert(fabs(-INFINITY) == INFINITY);
  __VERIFIER_assert(isnan(fabs(-NAN)));
}
