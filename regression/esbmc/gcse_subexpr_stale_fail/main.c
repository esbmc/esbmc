#include <assert.h>

unsigned nondet_uint();

// `(a + b) * c` recomputes `a + b` after `a` changed, but only its symbol is
// assigned, so the symbol for `a + b` must not be reused (#7992).
int main()
{
  unsigned a = nondet_uint(), b = nondet_uint(), c = nondet_uint();
  unsigned x = a + b;
  a = a + 1u;
  unsigned y = (a + b) * c;
  unsigned z = a + b;
  unsigned w = (a + b) * c;
  assert(z == x);
  return (int)(y + w);
}
