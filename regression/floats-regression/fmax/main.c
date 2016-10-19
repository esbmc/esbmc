#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void)
{
  __VERIFIER_assert(fmax(2,1) == 2.f);
  __VERIFIER_assert(fmax(-INFINITY,0) == 0);
  __VERIFIER_assert(fmax(NAN,-1) == -1.f);
  __VERIFIER_assert(!(fmax(NAN,NAN) == NAN));
}

