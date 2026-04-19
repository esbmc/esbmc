#include <assert.h>	
#include <math.h>

unsigned int nondet_uint();
double nondet_double();

int main() {
  unsigned int N = 2;
  double x = nondet_double();

  if(x <= 0 || isnan(x))
    return 0;

  unsigned int i = 0;

  __VERIFIER_assume(x < 5);

  while(i < N) {
    __VERIFIER_assume(x > 0);
    __VERIFIER_assume(0 <= i && i <= N);

    x = (2 * x);
    ++i;
  }

  assert(x > 0);
  return 0;
}
