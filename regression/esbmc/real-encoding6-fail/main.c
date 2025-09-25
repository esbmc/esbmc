#include <assert.h>
#include <math.h>

float nondet_float();

int main()
{
  float x = nondet_float();
  __VERIFIER_assume(isnan(x));  // Assume x is NaN
  assert(x == x);  // Should fail: NaN != NaN
  return 0;
}
