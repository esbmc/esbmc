#include <assert.h>
#include <math.h>

float nondet_float();

int main()
{
  float x = nondet_float();
  float y = nondet_float();
  
  __VERIFIER_assume(isnan(x));
  __VERIFIER_assume(!isnan(y));
  
  assert((x != x) && (y == y));  // Should pass: both are true
  return 0;
}
