#include <assert.h>
#include <math.h>
double nondet_double(void);
static double V(double c){ double x=nondet_double(); __ESBMC_assume(x==c); return x; }
int main(void)
{
  assert(remainder(V(6.0), V(4.0)) == -2.0);
  assert(fmod(V(6.0), V(4.0)) == 2.0);
  return 0;
}
