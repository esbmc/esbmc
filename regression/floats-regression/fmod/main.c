#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void)
{
  __VERIFIER_assert(fmodf(5.1f, 3) == 2.1f);
  __VERIFIER_assert(fmodf(-5.1f, 3) == -2.1f);
  __VERIFIER_assert(fmodf(5.1f, -3) == 2.1f);
  __VERIFIER_assert(fmodf(-5.1f, -3) == -2.1f);

  __VERIFIER_assert(fmodf(0, 1) == 0.0);
  __VERIFIER_assert(fmodf(-0.0, 1) == -0.0);
  __VERIFIER_assert(fmod(5.1, INFINITY) == 5.1);
}
