#include <math.h>
#include <float.h>
#include <inttypes.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void)
{
  double in = 0.1;
  __VERIFIER_assert(in == 0x1.999999999999ap-4);

  double expr_result = 0.1 * 10 - 1;
  __VERIFIER_assert(expr_result == 0.0);

  double fma_result = fma(0.1, 10, -1);
  __VERIFIER_assert(fma_result == 0x1p-54);

  __VERIFIER_assert(fma_result != expr_result);

  __VERIFIER_assert(isnan(fma(INFINITY, 10, -INFINITY)));
}

