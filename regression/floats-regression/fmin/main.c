#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void)
{
  __VERIFIER_assert(fmin(2,1) == 1.f);
  __VERIFIER_assert(fmin(-INFINITY,0) == -(1./0.0));
  __VERIFIER_assert(fmin(NAN,-1) == -1.f);
  __VERIFIER_assert(!(fmin(NAN,NAN) == NAN));
}

