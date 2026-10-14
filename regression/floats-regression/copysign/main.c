#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void)
{
  __VERIFIER_assert(copysign(1.0, +2.0) == 1.0);
  __VERIFIER_assert(copysign(1.0, -2.0) == -1.0);
  __VERIFIER_assert(copysign(-1.0, +2.0) == 1.0);
  __VERIFIER_assert(copysign(-1.0, -2.0) == -1.0);

  __VERIFIER_assert(copysign(INFINITY, -2.0) == -INFINITY);

  /* The sign of a NaN needs the bit-vector encoding (#7021); it is pinned by
     regression/bitwuzla/fp2bv_nan_sign. */
}

