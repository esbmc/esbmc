#include <assert.h>
#include <math.h>

float nondet_float();

int main()
{
  float x = nondet_float();
  __VERIFIER_assume(x == 1.0f); // not NaN
  assert(!isnan(x));
}

