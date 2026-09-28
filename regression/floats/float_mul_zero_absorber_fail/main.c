#include <assert.h>
#include <math.h>

float nondet_float(void);

int main()
{
  float i = nondet_float();
  __ESBMC_assume(isinf(i));
  assert(i * 0.0f == 0.0f);
  return 0;
}
