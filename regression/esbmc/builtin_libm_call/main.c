#include <assert.h>
#include <math.h>

double nondet_double(void);

int main(void)
{
  double y = nondet_double();
  __ESBMC_assume(!isnan(y));
  assert(__builtin_copysign(1.0, y) == (signbit(y) ? -1.0 : 1.0));
  assert(__builtin_fmax(y, 2.0) >= 2.0);

  int e = 0;
  assert(__builtin_frexp(8.0, &e) == 0.5 && e == 4);
  return 0;
}
