#include <assert.h>

float nondet_float();

int main()
{
  float x = nondet_float();
  __VERIFIER_assume(x > 0);
  assert(x != 0);
  return 0;
}
