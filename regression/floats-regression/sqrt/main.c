#include <math.h>

extern void __VERIFIER_assume(int);
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }
 
int main(void)
{
  __VERIFIER_assert(sqrt(100) == 10);
  __VERIFIER_assert(sqrt(2) == 1.41421356237309514547462185873883);
  __VERIFIER_assert(((1+sqrt(5))/2) == 1.61803398874989490252573887119070);
 
  double c1 = sqrt(-0.0);
  __VERIFIER_assert((c1 == -0.0) && signbit(c1));

  double c2 = sqrt(0.0);
  __VERIFIER_assert((c2 == 0.0) && !signbit(c2));

  double c3 = sqrt(INFINITY);
  __VERIFIER_assert(isinf(c3) && !signbit(c3));

  __VERIFIER_assert(isnan(sqrt(-INFINITY)));
  __VERIFIER_assert(isnan(sqrt(-1)));

  __VERIFIER_assert(sqrt(4) == 2.0);  
}

