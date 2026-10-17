/* esbmc/esbmc#8684: the selected -0.0 is negative, so both asserts fail. */
#include <assert.h>
#include <math.h>
int nondet_int(void);
int main(void)
{
  int c = nondet_int();
  double a = c ? -0.0 : 1.0;
  double b = c ? 1.0 : -0.0;
  assert(!c || !signbit(a));
  assert(c || !signbit(b));
  return 0;
}
