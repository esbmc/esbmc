#include <assert.h>
#include <stdio.h>
extern void __VERIFIER_assume(int cond);
extern float __VERIFIER_nondet_float(void);
int main() {
  float x = __VERIFIER_nondet_float();
  assert(x < 1.0  || x > -1.0);
  __VERIFIER_assume(x > 0.0);
  printf("%f\n", x);
  assert(x > 0.0);
  return 0;
}
