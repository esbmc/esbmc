#include <assert.h>
#include <math.h>
double nondet_double(void);
static double V(double c){ double x=nondet_double(); __ESBMC_assume(x==c); return x; }
int main(void)
{
  assert(V(1.0) + V(2.0) == 3.0);
  assert(V(1.0) - V(2.0) == -1.0);
  assert(V(3.0) * V(4.0) == 12.0);
  assert(V(1.0) / V(4.0) == 0.25);
  assert(sqrt(V(16.0)) == 4.0);
  assert(fma(V(2.0), V(3.0), V(1.0)) == 7.0);
  assert(fabs(V(-2.5)) == 2.5);
  assert(V(0.1) + V(0.2) != 0.3);
  assert(V(1.0) < V(2.0) && V(2.0) >= V(2.0));
  assert(isnan(V(0.0) / V(0.0)));
  assert(isinf(V(1.0) / V(0.0)));
  assert(nearbyint(V(2.5)) == 2.0);
  return 0;
}
