#include <assert.h>
#include <math.h>

float nondet_float();

int main()
{
  float x = nondet_float();
  __VERIFIER_assume(!isnan(x));  // Assume x is not NaN
  assert(x == x);  // Should pass: non-NaN == itself
  return 0;
}
