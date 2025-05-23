#include <assert.h>
#include <math.h>

float nondet_float();

int main()
{
  float x = nondet_float();
  assert(!isnan(x)); // x might be NaN
}

