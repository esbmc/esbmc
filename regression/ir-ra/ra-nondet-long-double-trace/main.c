#include <assert.h>

extern long double __VERIFIER_nondet_long_double(void);

int main(void)
{
  long double x = __VERIFIER_nondet_long_double();
  assert(x != x + 0.0L);
  return 0;
}
